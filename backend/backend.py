from typing import List, Optional, Tuple, Any

# Modal runs the entire script in every container.
# FastAPI is only installed in the web endpoint container.
# We wrap this in a try-except so worker containers don't crash on boot!
try:
    from fastapi import Request, Response
except ImportError:
    Request = Any
    Response = Any
import modal
import tempfile

temp_volume = modal.Volume.from_name("video-cache", create_if_missing=True)

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
ADAPTER_REPO_ID = "Playbird12/motioncoach-qwen3vl-32b-lora"
ADAPTER_CACHE_DIR = "/root/adapter_cache"
MODEL_CACHE_DIR = "/root/model_cache"


def download_model_weights():
    """Runs during image build to bake model weights into the image layer."""
    import os
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    snapshot_download(
        MODEL_ID,
        local_dir=MODEL_CACHE_DIR,
        token=token,
        # skip old-format weights, use safetensors only
        ignore_patterns=["*.pt", "*.bin"],
    )


def download_adapter_weights():
    """Runs during image build to bake LoRA adapter into the image layer."""
    import os
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")
    try:
        snapshot_download(
            ADAPTER_REPO_ID,
            local_dir=ADAPTER_CACHE_DIR,
            token=token,
        )
        print(f"[Build] LoRA adapter downloaded to {ADAPTER_CACHE_DIR}")
    except Exception as e:
        print(f"[Build] WARNING: Could not download LoRA adapter ({e}). "
              "Will fall back to base model at runtime.")


# ==============================================================================
# 🦴 POSE DETECTION HELPER FUNCTIONS
# ==============================================================================


# COCO skeleton connections (17 keypoints total)
COCO_SKELETON = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),  # Face/head
    (6, 12),
    (7, 13),
    (6, 7),  # Shoulders to hips
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),  # Arms
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),  # Torso/legs
]


def keypoints_to_vectors(
    xy: list,
    conf: list,
    width: int,
    height: int,
    min_conf: float = 0.15,  # Lowered from 0.25 to detect more poses
    color: str = "#888888",  # Gray for non-critiqued skeleton segments
) -> List[dict]:
    """Convert YOLO keypoints to app's vector format."""
    vectors = []

    for person_idx, person in enumerate(xy):
        for a, b in COCO_SKELETON:
            i = a - 1  # Convert to 0-indexed
            j = b - 1

            if i >= len(person) or j >= len(person):
                continue

            x1, y1 = person[i]
            x2, y2 = person[j]

            # Skip if confidence too low
            if conf is not None and (
                conf[person_idx][i] < min_conf or conf[person_idx][j] < min_conf
            ):
                continue

            # Normalize coordinates to 0-1 range
            vectors.append(
                {
                    "start": [float(x1 / width), float(y1 / height)],
                    "end": [float(x2 / width), float(y2 / height)],
                    "color": color,
                }
            )

    return vectors


def get_keypoint_coords(
    xy: list, conf: list, keypoint_idx: int, width: int, height: int
) -> Optional[Tuple[float, float]]:
    """Get normalized coordinates for a specific keypoint (0-indexed)."""
    if not xy or len(xy) == 0:
        return None

    person = xy[0]  # First person
    if keypoint_idx >= len(person):
        return None

    if conf and conf[0][keypoint_idx] < 0.15:
        return None

    x, y = person[keypoint_idx]
    return (float(x / width), float(y / height))


def snap_correction_vectors_to_skeleton(
    visuals: dict,
    xy: list,
    conf: list,
    width: int,
    height: int,
    coaching_script: str = "",
) -> dict:
    """
    Generate correction vectors anchored to actual detected keypoints.
    Uses ONLY the coaching script text to identify which body part needs correction,
    ignoring LLM's approximate coordinates entirely.
    """
    if not xy or len(xy) == 0:
        return visuals

    # COCO keypoint indices (0-indexed)
    KEYPOINTS = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }

    # Map keypoints to their connected neighbors in the skeleton
    KEYPOINT_CONNECTIONS = {
        "left_shoulder": ["left_elbow", "left_hip", "right_shoulder"],
        "right_shoulder": ["right_elbow", "right_hip", "left_shoulder"],
        "left_elbow": ["left_shoulder", "left_wrist"],
        "right_elbow": ["right_shoulder", "right_wrist"],
        "left_wrist": ["left_elbow"],
        "right_wrist": ["right_elbow"],
        "left_hip": ["left_shoulder", "left_knee", "right_hip"],
        "right_hip": ["right_shoulder", "right_knee", "left_hip"],
        "left_knee": ["left_hip", "left_ankle"],
        "right_knee": ["right_hip", "right_ankle"],
        "left_ankle": ["left_knee"],
        "right_ankle": ["right_knee"],
    }

    # Parse coaching script to identify EXACT body part (with side)
    script_lower = coaching_script.lower()

    # Detect side preference
    has_left = "left" in script_lower
    has_right = "right" in script_lower

    # Map body part keywords to keypoint names (SIDE-AWARE)
    BODY_PART_KEYWORDS = {
        "shoulder": {"left": ["left_shoulder"], "right": ["right_shoulder"]},
        "elbow": {"left": ["left_elbow"], "right": ["right_elbow"]},
        "wrist": {"left": ["left_wrist"], "right": ["right_wrist"]},
        "hand": {"left": ["left_wrist"], "right": ["right_wrist"]},
        "hip": {"left": ["left_hip"], "right": ["right_hip"]},
        "knee": {"left": ["left_knee"], "right": ["right_knee"]},
        "ankle": {"left": ["left_ankle"], "right": ["right_ankle"]},
        "foot": {"left": ["left_ankle"], "right": ["right_ankle"]},
        "arm": {
            "left": ["left_elbow", "left_shoulder"],
            "right": ["right_elbow", "right_shoulder"],
        },
        "leg": {
            "left": ["left_knee", "left_hip"],
            "right": ["right_knee", "right_hip"],
        },
    }

    target_keypoints = []
    for body_part, sides in BODY_PART_KEYWORDS.items():
        if body_part in script_lower:
            if has_left and not has_right:
                target_keypoints.extend(sides["left"])
                print(f"[Snap] Detected 'left {body_part}' -> {sides['left']}")
            elif has_right and not has_left:
                target_keypoints.extend(sides["right"])
                print(f"[Snap] Detected 'right {body_part}' -> {sides['right']}")
            else:
                # Ambiguous or both mentioned - include both sides
                target_keypoints.extend(sides["left"] + sides["right"])
                print(
                    f"[Snap] Detected '{body_part}' (both sides) -> {sides['left']} + {sides['right']}"
                )

    if not target_keypoints:
        print(
            f"[Snap] ⚠️  No specific body part found in script, skipping correction vectors"
        )
        return visuals

    # Generate correction vectors purely from detected keypoints
    snapped_vectors = []

    for kp_name in target_keypoints:
        if kp_name not in KEYPOINTS:
            continue

        kp_idx = KEYPOINTS[kp_name]
        kp_coords = get_keypoint_coords(xy, conf, kp_idx, width, height)

        if not kp_coords:
            print(f"[Snap] ⚠️  Keypoint {kp_name} not detected, skipping")
            continue

        # Find adjacent keypoint to form the limb segment
        if kp_name not in KEYPOINT_CONNECTIONS:
            continue

        adjacent_coords = None
        adjacent_name = None

        for adj_name in KEYPOINT_CONNECTIONS[kp_name]:
            if adj_name in KEYPOINTS:
                adj_idx = KEYPOINTS[adj_name]
                adj_coords = get_keypoint_coords(xy, conf, adj_idx, width, height)
                if adj_coords:
                    adjacent_coords = adj_coords
                    adjacent_name = adj_name
                    break

        if not adjacent_coords:
            print(f"[Snap] ⚠️  No adjacent keypoint for {kp_name}, skipping")
            continue

        # Vector 1: Current limb segment (RED) - shows what IS
        snapped_vectors.append(
            {
                "start": list(kp_coords),
                "end": list(adjacent_coords),
                "color": "#FF3B30",
                "label": "Current",
                "is_correction": True,
                "body_part": kp_name.replace("_", " ").title(),
            }
        )

        # Vector 2: Target position (GREEN) - shows what SHOULD BE
        # Use a rotation hint to show correction direction
        curr_dx = adjacent_coords[0] - kp_coords[0]
        curr_dy = adjacent_coords[1] - kp_coords[1]
        segment_length = (curr_dx**2 + curr_dy**2) ** 0.5

        # Rotate by 30 degrees as a visual correction hint
        import math

        angle = math.radians(30)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        target_x = kp_coords[0] + (curr_dx * cos_a - curr_dy * sin_a)
        target_y = kp_coords[1] + (curr_dx * sin_a + curr_dy * cos_a)

        snapped_vectors.append(
            {
                "start": list(kp_coords),
                "end": [target_x, target_y],
                "color": "#34C759",
                "label": "Target",
                "is_correction": True,
                "body_part": kp_name.replace("_", " ").title(),
            }
        )

        # Store the correction angle in the visuals so the frontend can label it
        # The angle is always 30 degrees (our rotation hint constant)
        if "correction_annotations" not in visuals:
            visuals["correction_annotations"] = []
        visuals["correction_annotations"].append(
            {
                "pivot": list(kp_coords),
                "body_part": kp_name.replace("_", " ").title(),
                "angle_deg": 30,
            }
        )

        print(
            f"[Snap] ✓ Generated correction pair for {kp_name} → {adjacent_name} (30° correction hint)"
        )

    # REPLACE all LLM vectors with our precisely-generated ones
    visuals["vectors"] = snapped_vectors

    print(
        f"[Snap] ✅ Replaced LLM vectors with {len(snapped_vectors)} precisely anchored correction vectors"
    )

    return visuals


# ==============================================================================


video_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .env({
        "REBUILD": "3",
        "CC": "gcc",
        "CUDA_HOME": "/usr/local/cuda",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "VLLM_USE_PRECOMPILED": "1",
    })
    .apt_install("git", "build-essential", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        "vllm",
        "qwen-vl-utils",
        "decord",
        "huggingface_hub",
        "ultralytics",
        "opencv-python-headless",
    )
    .run_function(
        download_model_weights,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .run_function(
        download_adapter_weights,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)

app = modal.App("biomechanics-ai")


@app.cls(
    gpu="H200:4",
    image=video_image,
    timeout=600,  # Max 10 mins per analysis
    scaledown_window=300,  # Keep container warm for 5 mins (faster subsequent requests)
    min_containers=1,  # Keep 1 container always ready (eliminates cold start)
    ephemeral_disk=512 * 1024,  # Minimum 512 GiB required by Modal for large models
    secrets=[modal.Secret.from_name("huggingface-secret")],
    enable_memory_snapshot=True,  # Snapshot GPU state after first load for fast restarts
)
class VideoAnalyzer:
    @modal.enter()
    def load_model(self):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        import time as _t
        import os

        print("[ModelLoad] Initializing vLLM with Qwen3-VL-32B + LoRA...")
        t0 = _t.time()
        self.llm = LLM(
            model=MODEL_CACHE_DIR,
            tensor_parallel_size=4,
            max_model_len=65536,
            gpu_memory_utilization=0.90,
            enable_lora=True,
            max_lora_rank=256,
        )
        self.sampling_params = SamplingParams(max_tokens=1536)

        if os.path.isdir(ADAPTER_CACHE_DIR) and os.listdir(ADAPTER_CACHE_DIR):
            self.lora_request = LoRARequest("motioncoach", 1, ADAPTER_CACHE_DIR)
            print(f"[ModelLoad] LoRA adapter loaded from {ADAPTER_CACHE_DIR}")
        else:
            self.lora_request = None
            print("[ModelLoad] No LoRA adapter found, using base model")

        print(
            f"[ModelLoad] vLLM ready in {_t.time() - t0:.1f}s — GPU snapshot will be taken"
        )

    @modal.method()
    def analyze(
        self,
        video_bytes: bytes,
        user_description: str,
        activity_type: str,
        previous_analysis: str = "",
        voice_id: str = "s3TPKV1kjDlVtZbl4Ksh",
        pose_data: Optional[dict] = None,
    ):
        import time as _time

        print(f"\n{'=' * 60}")
        print(f"[Analyze] 🎬 Starting VL analysis")
        print(f"[Analyze]   activity_type = {activity_type}")
        print(f"[Analyze]   user_description = {user_description}")
        print(f"[Analyze]   video_bytes = {len(video_bytes):,} bytes")
        print(f"[Analyze]   has_previous_analysis = {bool(previous_analysis)}")
        print(
            f"[Analyze]   has_pose_data = {pose_data is not None and bool(pose_data.get('detected_poses'))}"
        )
        t_start = _time.time()

        # === Activity-specific coaching context ===
        ACTIVITY_HINTS = {
            "basketball_shot": "Key biomechanics: elbow alignment under the ball, follow-through with wrist snap, balanced base with knees bent, shooting arc trajectory, guide hand placement.",
            "golf_swing": "Key biomechanics: hip rotation leading the downswing, maintaining spine angle, wrist hinge and release, weight transfer from back foot to front, club path and face angle at impact.",
            "tennis_serve": "Key biomechanics: trophy position with racket behind head, full extension at contact point, pronation of forearm, knee bend and explosive drive upward, toss placement.",
            "baseball_pitch": "Key biomechanics: stride length toward target, hip-shoulder separation, arm slot consistency, front foot landing, follow-through deceleration.",
            "soccer_kick": "Key biomechanics: plant foot placement beside the ball, hip rotation, locked ankle on striking foot, lean-back angle, follow-through direction.",
            "guitar_strumming": "Key biomechanics: wrist relaxation and pivot point, pick angle and grip pressure, elbow as the fulcrum, rhythm consistency, muting technique.",
            "piano_hands": "Key biomechanics: curved finger position, wrist height and relaxation, thumb passing technique, even key pressure, minimal unnecessary finger lift.",
            "dance_move": "Key biomechanics: center of gravity alignment, weight transfer timing, arm lines and extensions, head spotting for turns, rhythm synchronization.",
        }
        activity_hint = ACTIVITY_HINTS.get(
            activity_type,
            "Analyze the user's body mechanics, posture, and movement pattern for this activity.",
        )

        PERSONALITIES = {
            # Adam
            "s3TPKV1kjDlVtZbl4Ksh": "You are a supportive, human-centric coach who focuses on building a personal connection through storytelling. Give feedback that feels like a peer-to-peer conversation, emphasizing relatability and shared growth while maintaining a confident, grounded energy.",
            # Brock
            "DGzg6RaUqxGRTHSBjfgF": "You are a relentless, high-authority instructor who demands immediate results and zero excuses. Use short, barking commands and clipped sentences to push the user to their absolute limit with a loud, commanding cadence.",
            # Maria
            "vZzlAds9NzvLsFSWp0qk": "You are a calm, composed coach who provides feedback with emotional warmth and a gentle, steady rhythm. Focus on precise, clear instructions delivered with a smooth timbre that makes the user feel safe, capable, and mentally balanced.",
            # Anya
            "d3MFdIuCfbAIwiu7jC4a": "You are a fun, playful, and intelligent coach who brings a 'cool older sister' energy to every session. Use an approachable, cute tone with modern inflections to make the practice feel like a high-energy social hangout rather than a chore.",
            # Jon
            "Cz0K1kOv9tD8l0b5Qu53": "You are a relaxed, no-nonsense American coach who keeps things simple, natural, and easygoing. Avoid technical jargon or intense pressure; instead, offer clear, conversational advice that sounds like a casual chat over a backyard fence.",
        }
        personality = PERSONALITIES[voice_id]

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            # ── Build the coaching prompt ──
            is_retry = bool(previous_analysis)
            prompt = f"""You are MotionCoach, an AI movement coach.

Your personality:
{personality} Reference specific body parts and timestamps so feedback is precise.

Activity: {activity_type}
{activity_hint}

User's goal: {user_description}
"""
            if is_retry:
                prompt += f"""\n=== RETRY SESSION — COMPARING TO PREVIOUS ATTEMPT ===
This is NOT the user's first attempt. They are retrying the same movement to improve.

{previous_analysis}

INSTRUCTIONS FOR RETRY:
- Compare this attempt directly to the previous one above
- If they fixed a previous mistake, acknowledge it enthusiastically (e.g. "Great job fixing your elbow angle!")
- Do NOT repeat feedback for issues they already corrected
- Focus new feedback on the NEXT most impactful improvement they should make
- Do NOT include progress_score or improvement_delta — the backend computes these automatically
"""
            else:
                prompt += """\n=== NEW SESSION — FIRST ATTEMPT ===
This is the user's first attempt at this movement. There is no previous session to compare to.
- Do NOT include progress_score or improvement_delta — the backend computes these automatically
"""

            # Add detected pose skeleton information
            if pose_data and pose_data.get("detected_poses"):
                prompt += "\n=== DETECTED POSE KEYPOINTS ===\n"
                available_timestamps = []
                pose_items = list(pose_data["detected_poses"].items())
                for idx, (timestamp_ms, keypoints_data) in enumerate(pose_items):
                    available_timestamps.append(timestamp_ms)
                    prompt += f"{timestamp_ms}ms: "
                    if keypoints_data.get("keypoints"):
                        key_joints = [
                            "left_shoulder",
                            "right_shoulder",
                            "left_elbow",
                            "right_elbow",
                            "left_wrist",
                            "right_wrist",
                            "left_hip",
                            "right_hip",
                            "left_knee",
                            "right_knee",
                        ]
                        kp_data = keypoints_data["keypoints"]
                        coords_str = ", ".join(
                            [
                                f"{kp}=[{kp_data[kp][0]:.2f},{kp_data[kp][1]:.2f}]"
                                for kp in key_joints
                                if kp in kp_data
                            ]
                        )
                        prompt += coords_str + "\n"

                print(
                    f"[Analyze] 📊 Pose data: {len(available_timestamps)} frames"
                )

                prompt += f"""\nTIMESTAMP RULES: Use exact timestamps from: {", ".join(map(str, available_timestamps[:10]))}{"..." if len(available_timestamps) > 10 else ""}
VISUAL RULES: Backend draws skeleton automatically. Always specify LEFT/RIGHT body parts in coaching_script.
"""

            prompt += """
=== OUTPUT FORMAT (strict JSON, no markdown fences) ===

You MUST return ONLY a JSON object with EXACTLY these top-level keys:
  "status": "success",
  "positive_note": "<one encouraging sentence about their overall form>",
  "feedback_points": [ ... ]

Each object in "feedback_points" MUST have EXACTLY these keys (no extras):
  "mistake_timestamp_ms": <int from the available timestamps>,
  "severity": "major" | "intermediate" | "minor",
  "positive_note": "<one encouraging sentence specific to this moment>",
  "coaching_script": "<the correction advice, specify LEFT/RIGHT body parts>",
  "visuals": {
    "overlay_type": "ANGLE_CORRECTION" | "POSITION_MARKER" | "PATH_TRACE",
    "focus_point": {"x": <float 0-1>, "y": <float 0-1>},
    "vectors": [{"start": [x,y], "end": [x,y], "color": "red"|"green", "label": "Current"|"Target"}],
    "path_points": null
  }

RULES:
- Return 1-5 feedback_points. Mix severities: use "major" for serious form issues, "intermediate" for moderate, "minor" for small tweaks.
- DO NOT omit "severity" or "coaching_script" — every feedback point MUST have both.
- DO NOT say timestamps in coaching_script (no "at 0.3s" or "at 1500ms").
- DO NOT add keys not listed above (no "feedback_script", "progress_delta", "improvement_delta", etc.).
- Prioritize: major issues first, then intermediate, then minor.

Example:
{"status":"success","positive_note":"Good form!","feedback_points":[{"mistake_timestamp_ms":1200,"severity":"major","positive_note":"Nice power in your swing!","coaching_script":"Raise your LEFT elbow higher to improve your arc.","visuals":{"overlay_type":"ANGLE_CORRECTION","focus_point":{"x":0.5,"y":0.5},"vectors":[{"start":[0.5,0.5],"end":[0.6,0.6],"color":"red","label":"Current"},{"start":[0.5,0.5],"end":[0.4,0.4],"color":"green","label":"Target"}],"path_points":null}},{"mistake_timestamp_ms":2400,"severity":"intermediate","positive_note":"Great rotation here!","coaching_script":"Keep your head more neutral through the motion.","visuals":{"overlay_type":"POSITION_MARKER","focus_point":{"x":0.5,"y":0.2},"vectors":[],"path_points":null}},{"mistake_timestamp_ms":3000,"severity":"minor","positive_note":"Solid base position.","coaching_script":"Slight RIGHT shoulder adjustment needed.","visuals":{"overlay_type":"POSITION_MARKER","focus_point":{"x":0.65,"y":0.35},"vectors":[],"path_points":null}}]}
"""

            print(f"\n[Analyze] 📝 Prompt constructed ({len(prompt):,} chars)")
            print(f"[Analyze] --- PROMPT START ---")
            print(prompt[:2000])
            if len(prompt) > 2000:
                print(f"  ... ({len(prompt) - 2000} more chars) ...")
            print(f"[Analyze] --- PROMPT END ---")

            # --- Frame extraction with decord ---
            import numpy as np
            from decord import VideoReader, cpu as decord_cpu
            from PIL import Image as PILImage

            vr = VideoReader(tmp.name, ctx=decord_cpu(0))
            actual_fps = vr.get_avg_fps()
            total_frames = len(vr)
            total_duration = total_frames / actual_fps

            # Sample at 8 fps for the first 5 seconds
            end_time = min(5.0, total_duration)
            nframes = max(1, int(end_time * 8.0))
            frame_indices = np.linspace(
                0,
                min(int(end_time * actual_fps), total_frames) - 1,
                num=nframes,
                dtype=int,
            ).tolist()
            raw_frames = vr.get_batch(frame_indices).asnumpy()

            MAX_DIM = 480
            video_frames = []
            for f in raw_frames:
                img = PILImage.fromarray(f)
                w, h = img.size
                if max(w, h) > MAX_DIM:
                    scale = MAX_DIM / max(w, h)
                    img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
                video_frames.append(img)
            del vr

            print(
                f"[Analyze] 🎞️  Extracted {len(video_frames)} frames from {total_duration:.1f}s video "
                f"({actual_fps:.1f} native fps, sampled {nframes} frames over {end_time:.1f}s, max {MAX_DIM}px)"
            )

            import base64
            import io

            content = []
            for frame in video_frames:
                buf = io.BytesIO()
                frame.save(buf, format="JPEG", quality=75)
                b64 = base64.b64encode(buf.getvalue()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]

            print(f"[Analyze] 🧠 Built {len(video_frames)} base64 frames for vLLM chat")

            t_gen_start = _time.time()
            print(f"[Analyze] 🚀 Starting llm.chat (max_tokens=1536)...")
            outputs = self.llm.chat(
                messages, self.sampling_params, lora_request=self.lora_request
            )
            t_gen_end = _time.time()
            output_text = outputs[0].outputs[0].text

            print(f"[Analyze] Generation complete in {t_gen_end - t_gen_start:.1f}s")
            print(f"[Analyze] --- RAW MODEL OUTPUT ({len(output_text)} chars) ---")
            print(output_text[:3000])
            if len(output_text) > 3000:
                print(f"  ... ({len(output_text) - 3000} more chars) ...")
            print(f"[Analyze] --- RAW MODEL OUTPUT END ---")

            # Clean markdown code blocks if the LLM added them
            import re
            import json as json_lib

            json_match = re.search(r"```json\n*(.*?)\n*```", output_text, re.DOTALL)
            clean_json_str = json_match.group(1) if json_match else output_text.strip()

            # Parse JSON response
            result = json_lib.loads(clean_json_str)

            # ── Normalize raw model output into frontend-compatible schema ────
            raw_fps = result.get("feedback_points", [])[:5]
            if len(result.get("feedback_points", [])) > 5:
                print(f"[Analyze] Capping feedback_points from {len(result['feedback_points'])} to 5")

            clean_fps = []
            for raw_fp in raw_fps:
                script = (
                    raw_fp.get("coaching_script")
                    or raw_fp.get("feedback_script")
                    or ""
                )
                if not script:
                    print(f"[Analyze] ⚠️  Skipping feedback point with no coaching text")
                    continue

                severity = raw_fp.get("severity", "intermediate")
                if severity not in ("major", "intermediate", "minor"):
                    print(f"[Analyze] ⚠️  Invalid severity '{severity}' → defaulting to 'intermediate'")
                    severity = "intermediate"

                raw_visuals = raw_fp.get("visuals")
                clean_visuals = None
                if raw_visuals and isinstance(raw_visuals, dict):
                    overlay = raw_visuals.get("overlay_type", "ANGLE_CORRECTION")
                    if overlay not in ("ANGLE_CORRECTION", "POSITION_MARKER", "PATH_TRACE", "POINT_HIGHLIGHT"):
                        overlay = "ANGLE_CORRECTION"

                    clean_vectors = []
                    for v in raw_visuals.get("vectors") or []:
                        if isinstance(v, dict) and "start" in v and "end" in v:
                            clean_vectors.append({
                                "start": list(v["start"])[:2],
                                "end": list(v["end"])[:2],
                                "color": v.get("color", "red"),
                                **({"label": v["label"]} if v.get("label") else {}),
                            })

                    clean_visuals = {
                        "overlay_type": overlay,
                        **({"focus_point": raw_visuals["focus_point"]} if raw_visuals.get("focus_point") else {}),
                        "vectors": clean_vectors,
                        **({"path_points": raw_visuals["path_points"]} if raw_visuals.get("path_points") else {}),
                    }

                fp_out = {
                    "mistake_timestamp_ms": int(raw_fp.get("mistake_timestamp_ms", 0)),
                    "coaching_script": script,
                    "severity": severity,
                    "visuals": clean_visuals,
                    "audio_url": "",
                }
                pos_note = (raw_fp.get("positive_note") or "").strip()
                if pos_note:
                    fp_out["positive_note"] = pos_note
                clean_fps.append(fp_out)

            clean_fps.sort(key=lambda fp: fp["mistake_timestamp_ms"])
            print(f"[Analyze] ✓ Normalized {len(clean_fps)} feedback points (sorted by timestamp)")

            # ── Deterministic score formula ─────────────────────────────────────
            SEVERITY_PENALTIES = {"major": 12, "intermediate": 5, "minor": 2}
            n_major = sum(1 for fp in clean_fps if fp["severity"] == "major")
            n_intermediate = sum(1 for fp in clean_fps if fp["severity"] == "intermediate")
            n_minor = sum(1 for fp in clean_fps if fp["severity"] == "minor")
            raw_score = (
                100
                - n_major * SEVERITY_PENALTIES["major"]
                - n_intermediate * SEVERITY_PENALTIES["intermediate"]
                - n_minor * SEVERITY_PENALTIES["minor"]
            )
            computed_score = max(10, min(100, raw_score))

            improvement_delta = None
            prev_score = None
            if previous_analysis:
                try:
                    prev = json_lib.loads(previous_analysis)
                    prev_score = prev.get("progress_score")
                    if prev_score is not None:
                        improvement_delta = computed_score - int(prev_score)
                except Exception:
                    pass

            result = {
                "status": result.get("status", "success"),
                "positive_note": result.get("positive_note", ""),
                "progress_score": computed_score,
                "improvement_delta": improvement_delta,
                "feedback_points": clean_fps,
            }
            if result.get("status") == "error":
                result["error_message"] = result.get("error_message", "Unknown error")

            print(
                f"[Analyze] 📐 Score: 100 - {n_major}×12 - {n_intermediate}×5 - {n_minor}×2"
                f" = {raw_score} → clamped → {computed_score}"
            )
            if improvement_delta is not None:
                print(f"[Analyze] 📈 improvement_delta = {improvement_delta:+d} (prev={prev_score}, curr={computed_score})")
            else:
                print(f"[Analyze] 📈 improvement_delta = null (first session)")

            t_end = _time.time()
            print(f"\n[Analyze] 📊 Parsed result summary:")
            print(f"  status: {result['status']}")
            print(f"  progress_score: {result['progress_score']}")
            print(f"  improvement_delta: {result['improvement_delta']}")
            print(f"  positive_note: {result['positive_note'][:100]}")
            print(f"  feedback_points: {len(result['feedback_points'])}")
            for i, fp in enumerate(result["feedback_points"]):
                print(
                    f"    [{i}] @{fp['mistake_timestamp_ms']}ms [{fp['severity']}]: {fp['coaching_script'][:60]}..."
                )
            print(
                f"[Analyze] ⏱️  Total analyze time: {t_end - t_start:.1f}s (generation: {t_gen_end - t_gen_start:.1f}s)"
            )
            print(f"{'=' * 60}\n")

        return result

    @modal.method()
    def extract_pose_data(self, video_bytes: bytes, timestamps_ms: List[int]) -> dict:
        """Extract pose keypoints at specific timestamps for LLM context using batch inference."""
        import cv2
        import numpy as np
        import time as _time
        from ultralytics import YOLO

        print(f"[Pose] Extracting pose data at {len(timestamps_ms)} timestamps...")
        t_start = _time.time()

        model = YOLO("yolo26x-pose.pt")

        # COCO keypoint names for clarity
        KEYPOINT_NAMES = [
            "nose",
            "left_eye",
            "right_eye",
            "left_ear",
            "right_ear",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]

        detected_poses = {}

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                return {"detected_poses": {}}

            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Convert timestamps to frame indices
            frame_indices = [int((ts / 1000.0) * fps) for ts in timestamps_ms]

            # Read all frames at once (much faster than seeking)
            frames = []
            frame_to_ts = {}

            for idx, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap.read()
                if success and frame is not None:
                    frames.append(frame)
                    frame_to_ts[len(frames) - 1] = timestamps_ms[idx]

            cap.release()

            if not frames:
                print("[Pose] ⚠️  No frames could be read")
                return {"detected_poses": {}}

            print(f"[Pose] Read {len(frames)} frames in {_time.time() - t_start:.1f}s")
            t_inference_start = _time.time()

            # BATCH INFERENCE - process all frames at once (MUCH faster on GPU)
            results = model(frames, verbose=False, conf=0.15)

            print(
                f"[Pose] Batch inference on {len(frames)} frames took {_time.time() - t_inference_start:.1f}s"
            )

            # Parse results
            for frame_idx, result in enumerate(results):
                ts = frame_to_ts[frame_idx]
                keypoints = result.keypoints

                if keypoints is None or keypoints.xy is None:
                    continue

                xy = keypoints.xy.cpu().numpy().tolist()
                conf = keypoints.conf
                conf = conf.cpu().numpy().tolist() if conf is not None else None

                # Extract first person's keypoints with names
                if len(xy) > 0:
                    person = xy[0]
                    person_conf = conf[0] if conf else [1.0] * len(person)

                    keypoints_dict = {}
                    for idx, (x, y) in enumerate(person):
                        if idx < len(KEYPOINT_NAMES) and person_conf[idx] > 0.15:
                            keypoints_dict[KEYPOINT_NAMES[idx]] = [
                                round(x / width, 3),
                                round(y / height, 3),
                            ]

                    detected_poses[str(ts)] = {
                        "num_people": len(xy),
                        "keypoints": keypoints_dict,
                    }

            print(
                f"[Pose] ✅ Extracted poses from {len(detected_poses)}/{len(frames)} frames in {_time.time() - t_start:.1f}s total"
            )
            return {"detected_poses": detected_poses}

    @modal.method()
    def add_pose_overlays(
        self, video_bytes: bytes, feedback_points: List[dict]
    ) -> List[dict]:
        """Extract pose skeleton for each feedback timestamp and merge with visuals."""
        import cv2
        from ultralytics import YOLO

        print(f"[Pose] Loading YOLO model...")
        model = YOLO("yolo26x-pose.pt")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                print("[Pose] ERROR: Could not open video")
                return feedback_points

            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            print(
                f"[Pose] Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames"
            )

            for fp_idx, fp in enumerate(feedback_points):
                timestamp_ms = fp.get("mistake_timestamp_ms", 0)
                visuals = fp.get("visuals")
                coaching_script = fp.get("coaching_script", "")

                # If no visuals exist, create a basic structure for the skeleton
                if not visuals:
                    print(f"[Pose] Creating empty visuals for feedback point {fp_idx}")
                    visuals = {
                        "overlay_type": "POSITION_MARKER",
                        "focus_point": None,
                        "vectors": [],
                    }
                    fp["visuals"] = visuals

                if not timestamp_ms:
                    print(f"[Pose] Skipping feedback point {fp_idx}: no timestamp")
                    continue

                try:
                    # Try the exact timestamp first
                    timestamps_to_try = [
                        timestamp_ms,
                        max(0, timestamp_ms - 100),  # 100ms before
                        timestamp_ms + 100,  # 100ms after
                    ]

                    pose_vectors = []
                    for ts in timestamps_to_try:
                        cap.set(cv2.CAP_PROP_POS_MSEC, ts)
                        success, frame = cap.read()

                        if not success or frame is None:
                            continue

                        # Run pose detection
                        results = model(
                            frame, verbose=False, conf=0.15
                        )  # Lower confidence

                        if not results or len(results) == 0:
                            continue

                        result = results[0]
                        keypoints = result.keypoints

                        if keypoints is None or keypoints.xy is None:
                            continue

                        xy = keypoints.xy.cpu().numpy().tolist()
                        conf = keypoints.conf
                        conf = conf.cpu().numpy().tolist() if conf is not None else None

                        # Convert to app vector format (gray skeleton by default)
                        pose_vectors = keypoints_to_vectors(
                            xy, conf, width, height, min_conf=0.15
                        )

                        if pose_vectors:
                            print(
                                f"[Pose] ✓ Found {len(pose_vectors)} skeleton vectors at {ts}ms for feedback {fp_idx}"
                            )

                            # Snap any existing correction vectors to detected skeleton for better alignment
                            if visuals.get("vectors"):
                                print(
                                    f"[Pose] Snapping {len(visuals['vectors'])} correction vectors to skeleton..."
                                )
                                visuals = snap_correction_vectors_to_skeleton(
                                    visuals, xy, conf, width, height, coaching_script
                                )

                            # Recolor skeleton segments that touch critiqued body parts
                            # Side-aware: "left elbow" only highlights left arm, not both
                            critiqued_kp_indices = set()
                            script_lower = coaching_script.lower()

                            # Detect side preference from coaching script
                            has_left = "left" in script_lower
                            has_right = "right" in script_lower

                            # Side-aware body part keyword mapping (COCO 0-indexed)
                            # Each entry: keyword -> {"left": [...], "right": [...], "both": [...]}
                            BODY_PART_KW = {
                                "shoulder": {"left": [5], "right": [6], "both": [5, 6]},
                                "elbow": {"left": [7], "right": [8], "both": [7, 8]},
                                "arm": {
                                    "left": [5, 7, 9],
                                    "right": [6, 8, 10],
                                    "both": [5, 6, 7, 8, 9, 10],
                                },
                                "wrist": {"left": [9], "right": [10], "both": [9, 10]},
                                "hand": {"left": [9], "right": [10], "both": [9, 10]},
                                "hip": {"left": [11], "right": [12], "both": [11, 12]},
                                "knee": {"left": [13], "right": [14], "both": [13, 14]},
                                "leg": {
                                    "left": [11, 13, 15],
                                    "right": [12, 14, 16],
                                    "both": [11, 12, 13, 14, 15, 16],
                                },
                                "ankle": {
                                    "left": [15],
                                    "right": [16],
                                    "both": [15, 16],
                                },
                                "foot": {"left": [15], "right": [16], "both": [15, 16]},
                                "back": {
                                    "left": [5, 6, 11, 12],
                                    "right": [5, 6, 11, 12],
                                    "both": [5, 6, 11, 12],
                                },
                                "spine": {
                                    "left": [5, 6, 11, 12],
                                    "right": [5, 6, 11, 12],
                                    "both": [5, 6, 11, 12],
                                },
                                "head": {
                                    "left": [0, 3, 4],
                                    "right": [0, 3, 4],
                                    "both": [0, 3, 4],
                                },
                                "neck": {
                                    "left": [0, 5, 6],
                                    "right": [0, 5, 6],
                                    "both": [0, 5, 6],
                                },
                            }
                            for kw, sides in BODY_PART_KW.items():
                                if kw in script_lower:
                                    # Pick the right side based on what the script mentions
                                    if has_left and not has_right:
                                        critiqued_kp_indices.update(sides["left"])
                                    elif has_right and not has_left:
                                        critiqued_kp_indices.update(sides["right"])
                                    else:
                                        critiqued_kp_indices.update(sides["both"])

                            if critiqued_kp_indices:
                                print(
                                    f"[Pose] 🎯 Critiqued keypoint indices: {sorted(critiqued_kp_indices)}"
                                )
                                # Build a set of (kp_a, kp_b) pairs that should be red
                                # EITHER endpoint touching a critiqued keypoint turns the segment red
                                critiqued_segments = set()
                                for a, b in COCO_SKELETON:
                                    i_a, i_b = a - 1, b - 1  # 0-indexed
                                    if (
                                        i_a in critiqued_kp_indices
                                        or i_b in critiqued_kp_indices
                                    ):
                                        critiqued_segments.add((i_a, i_b))

                                # Recolor matching gray skeleton vectors to red (#FF3B30)
                                recolored = 0
                                person = xy[0] if xy else []
                                for vec in pose_vectors:
                                    if vec.get("color") != "#888888":
                                        continue
                                    vs = vec["start"]
                                    ve = vec["end"]
                                    for i_a, i_b in critiqued_segments:
                                        if i_a >= len(person) or i_b >= len(person):
                                            continue
                                        sx, sy = (
                                            float(person[i_a][0] / width),
                                            float(person[i_a][1] / height),
                                        )
                                        ex, ey = (
                                            float(person[i_b][0] / width),
                                            float(person[i_b][1] / height),
                                        )
                                        if (
                                            abs(vs[0] - sx) < 0.01
                                            and abs(vs[1] - sy) < 0.01
                                            and abs(ve[0] - ex) < 0.01
                                            and abs(ve[1] - ey) < 0.01
                                        ):
                                            vec["color"] = "#FF3B30"
                                            recolored += 1
                                            break
                                print(
                                    f"[Pose] 🎨 Recolored {recolored}/{len(critiqued_segments)} critiqued skeleton segments to red"
                                )

                            # Merge skeleton with (now-snapped) correction vectors
                            existing_vectors = visuals.get("vectors") or []
                            combined_vectors = pose_vectors + existing_vectors
                            visuals["vectors"] = combined_vectors

                            print(
                                f"[Pose] Combined {len(pose_vectors)} skeleton + {len(existing_vectors)} correction vectors"
                            )
                            break
                        else:
                            print(f"[Pose] No valid keypoints at {ts}ms")

                    if not pose_vectors:
                        print(
                            f"[Pose] ✗ No pose detected for feedback {fp_idx} at {timestamp_ms}ms"
                        )
                        continue

                except Exception as e:
                    print(f"[Pose] ERROR at {timestamp_ms}ms: {e}")
                    import traceback

                    traceback.print_exc()

            cap.release()

        return feedback_points


@app.function(
    image=modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("fastapi", "python-multipart", "opencv-python", "elevenlabs"),
    volumes={"/tmp/video-cache": temp_volume},
    secrets=[modal.Secret.from_name("custom-secret")],
    timeout=600,
    scaledown_window=120,  # Keep endpoint warm for 2 mins
    min_containers=1,  # Always have 1 endpoint container ready
)
@modal.fastapi_endpoint(method="POST")
async def analyze(request: Request):
    import json
    import time as _time
    import os
    import base64
    from elevenlabs.client import ElevenLabs

    t_request_start = _time.time()

    # 1. Parse Multipart Form Data
    form = await request.form()

    video_file = form.get("video_file")
    activity_type = form.get("activity_type", "Unknown Activity")
    user_description = form.get("user_description", "")
    previous_analysis_str = form.get("previous_analysis", "")
    elevenlabs_voice_id = form.get("voice_id", "s3TPKV1kjDlVtZbl4Ksh")
    print(f"\n{'#' * 60}")
    print(f"[Endpoint] 📥 Incoming POST /analyze")
    print(f"[Endpoint]   activity_type = {activity_type}")
    print(f"[Endpoint]   user_description = {user_description[:100]}")
    print(f"[Endpoint]   has_previous_analysis = {bool(previous_analysis_str)}")
    print(f"[Endpoint]   has_video_file = {video_file is not None}")

    if not video_file:
        print(f"[Endpoint] ❌ No video_file in form data")
        return Response(
            content=json.dumps(
                {
                    "status": "error",
                    "error_message": "Missing video_file in form data.",
                    "positive_note": "",
                    "progress_score": 0,
                    "feedback_points": [],
                }
            ),
            media_type="application/json",
        )

    video_bytes = await video_file.read()
    print(
        f"[Endpoint]   video_size = {len(video_bytes):,} bytes ({len(video_bytes) / 1024 / 1024:.1f} MB)"
    )

    # 2. Extract pose data from video for LLM context
    # Sample every 6th frame for optimal speed/accuracy balance
    try:
        import cv2
        import tempfile as tf

        # Determine video fps and total frames to calculate frame timestamps
        with tf.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            cap = cv2.VideoCapture(tmp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_seconds = total_frames / fps
            cap.release()

            # Sample at ~4 poses/sec for the first 5 seconds
            max_duration = min(5.0, duration_seconds)
            sample_fps = min(fps, 4.0)
            n_samples = max(1, int(max_duration * sample_fps))
            sample_timestamps = [
                int((i / sample_fps) * 1000) for i in range(n_samples)
            ]
            print(
                f"[Endpoint] Video: {fps:.2f} fps, {total_frames} frames total, {duration_seconds:.1f}s duration"
            )
            print(
                f"[Endpoint] 🦴 Extracting pose at {sample_fps:.0f} fps → {len(sample_timestamps)} samples for first {max_duration:.1f}s"
            )
            print(
                f"[Endpoint] 🦴 Generated {len(sample_timestamps)} timestamps: {sample_timestamps[:10]}{'...' if len(sample_timestamps) > 10 else ''}"
            )

        t_pose_start = _time.time()
        pose_data = await VideoAnalyzer().extract_pose_data.remote.aio(
            video_bytes=video_bytes, timestamps_ms=sample_timestamps
        )
        t_pose_end = _time.time()
        n_poses = len(pose_data.get("detected_poses", {}))
        print(
            f"[Endpoint] ✅ Pose data extracted: {n_poses} frames in {t_pose_end - t_pose_start:.1f}s"
        )
    except Exception as e:
        print(
            f"[Endpoint] ⚠️  Pose extraction failed, continuing without pose context: {e}"
        )
        pose_data = None

    # 3. Call the VL model with pose context
    try:
        t_llm_start = _time.time()
        print(f"[Endpoint] 🧠 Calling VideoAnalyzer.analyze...")
        llm_response = await VideoAnalyzer().analyze.remote.aio(
            video_bytes=video_bytes,
            user_description=user_description,
            activity_type=activity_type,
            previous_analysis=previous_analysis_str,
            pose_data=pose_data,
            voice_id=elevenlabs_voice_id,
        )
        t_llm_end = _time.time()
        print(f"[Endpoint] ✅ VL analysis complete in {t_llm_end - t_llm_start:.1f}s")
        print(
            f"[Endpoint]   LLM status={llm_response.get('status')}, score={llm_response.get('progress_score')}, feedback_points={len(llm_response.get('feedback_points', []))}"
        )
    except Exception as e:
        print(f"[Endpoint] ❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return Response(
            content=json.dumps(
                {
                    "status": "error",
                    "error_message": f"Analysis failed: {str(e)}",
                    "positive_note": "",
                    "progress_score": 0,
                    "feedback_points": [],
                }
            ),
            media_type="application/json",
        )

    if llm_response.get("status") == "error":
        print(
            f"[Endpoint] ❌ Model returned error: {llm_response.get('error_message')}"
        )
        return Response(content=json.dumps(llm_response), media_type="application/json")

    # 4. Generate TTS audio for each feedback point
    feedback_points = llm_response.get("feedback_points", [])
    audio_results = [""] * len(feedback_points)

    if feedback_points:
        try:
            t_tts_start = _time.time()
            api_key = os.environ.get("elevenlabs")
            if not api_key:
                print(
                    "[Endpoint] ⚠️  ElevenLabs API key not found, continuing without audio"
                )
                raise RuntimeError("missing_elevenlabs_key")

            client = ElevenLabs(api_key=api_key)
            print(
                f"[Endpoint] 🔊 Generating ElevenLabs audio for {len(feedback_points)} feedback points..."
            )

            for idx, fp in enumerate(feedback_points):
                script = (fp.get("coaching_script") or "").strip()
                if not script:
                    continue

                audio_gen = client.text_to_speech.convert(
                    text=script,
                    voice_id=elevenlabs_voice_id,
                )
                audio_bytes = b"".join(list(audio_gen))
                audio_results[idx] = base64.b64encode(audio_bytes).decode("utf-8")

            t_tts_end = _time.time()
            n_generated = sum(1 for a in audio_results if a)
            print(
                f"[Endpoint] ✅ Generated {n_generated}/{len(feedback_points)} audio clips in {t_tts_end - t_tts_start:.1f}s"
            )
        except Exception as e:
            if str(e) != "missing_elevenlabs_key":
                print(
                    f"[Endpoint] ⚠️  ElevenLabs TTS failed ({type(e).__name__}, {e}), continuing without audio"
                )

    # 5. Add pose skeleton overlays to each feedback timestamp
    try:
        t_overlay_start = _time.time()
        print(
            f"[Endpoint] 🦴 Adding pose skeleton overlays to {len(feedback_points)} feedback points..."
        )
        feedback_points = await VideoAnalyzer().add_pose_overlays.remote.aio(
            video_bytes=video_bytes, feedback_points=feedback_points
        )
        t_overlay_end = _time.time()
        print(
            f"[Endpoint] ✅ Pose skeleton overlays added in {t_overlay_end - t_overlay_start:.1f}s"
        )
    except Exception as e:
        print(f"[Endpoint] ⚠️  Pose overlay generation failed: {e}")
        import traceback

        traceback.print_exc()

    # 6. Construct Final Response
    final_feedback_points = []
    for fp, audio_b64 in zip(feedback_points, audio_results):
        fp_copy = fp.copy()
        fp_copy["audio_url"] = (
            f"data:audio/mpeg;base64,{audio_b64}" if audio_b64 else ""
        )
        final_feedback_points.append(fp_copy)

    llm_response["feedback_points"] = final_feedback_points

    # Log the final JSON payload (truncate audio/video to avoid log spam)
    t_request_end = _time.time()
    print(f"\n[Endpoint] 📤 FINAL RESPONSE PAYLOAD:")
    log_response = json.loads(json.dumps(llm_response))  # deep copy
    for fp in log_response.get("feedback_points", []):
        if "audio_url" in fp:
            fp["audio_url"] = fp["audio_url"][:50] + "...[TRUNCATED]"
        if "video_url" in fp:
            fp["video_url"] = fp["video_url"][:50] + "...[TRUNCATED]"
        # Truncate vectors list for readability
        if fp.get("visuals") and fp["visuals"].get("vectors"):
            n_vecs = len(fp["visuals"]["vectors"])
            if n_vecs > 3:
                fp["visuals"]["vectors"] = fp["visuals"]["vectors"][:3]
                fp["visuals"]["vectors"].append(
                    {"_truncated": f"...and {n_vecs - 3} more vectors"}
                )
    print(json.dumps(log_response, indent=2))
    print(f"\n[Endpoint] ⏱️  Total request time: {t_request_end - t_request_start:.1f}s")
    print(f"{'#' * 60}\n")

    return Response(content=json.dumps(llm_response), media_type="application/json")
