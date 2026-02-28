from typing import List, Optional, Literal, Tuple, Any

# Modal runs the entire script in every container.
# FastAPI is only installed in the web endpoint container.
# We wrap this in a try-except so worker containers don't crash on boot!
try:
    from fastapi import Request, Response
except ImportError:
    Request = Any
    Response = Any
import modal
from pydantic import BaseModel
import io
import base64
import tempfile

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct"
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

# ==============================================================================
# 🦴 POSE DETECTION HELPER FUNCTIONS
# ==============================================================================


# COCO skeleton connections (17 keypoints total)
COCO_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),  # Face/head
    (6, 12), (7, 13), (6, 7),  # Shoulders to hips
    (6, 8), (7, 9), (8, 10), (9, 11),  # Arms
    (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),  # Torso/legs
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
            if conf is not None and (conf[person_idx][i] < min_conf or conf[person_idx][j] < min_conf):
                continue

            # Normalize coordinates to 0-1 range
            vectors.append({
                "start": [float(x1 / width), float(y1 / height)],
                "end": [float(x2 / width), float(y2 / height)],
                "color": color
            })

    return vectors


def get_keypoint_coords(xy: list, conf: list, keypoint_idx: int, width: int, height: int) -> Optional[Tuple[float, float]]:
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
    coaching_script: str = ""
) -> dict:
    """
    Snap correction vectors to actual detected keypoints for better alignment.
    Uses the coaching script to identify which body part is being corrected.
    """
    if not xy or len(xy) == 0:
        return visuals

    # COCO keypoint indices (0-indexed)
    KEYPOINTS = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
    }

    # Map body part mentions to keypoint groups
    BODY_PART_KEYWORDS = {
        "back": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "spine": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "shoulder": ["left_shoulder", "right_shoulder"],
        "elbow": ["left_elbow", "right_elbow"],
        "arm": ["left_shoulder", "left_elbow", "left_wrist", "right_shoulder", "right_elbow", "right_wrist"],
        "wrist": ["left_wrist", "right_wrist"],
        "hand": ["left_wrist", "right_wrist"],
        "hip": ["left_hip", "right_hip"],
        "knee": ["left_knee", "right_knee"],
        "leg": ["left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle"],
        "ankle": ["left_ankle", "right_ankle"],
        "foot": ["left_ankle", "right_ankle"],
        "head": ["nose", "left_ear", "right_ear"],
        "neck": ["nose", "left_shoulder", "right_shoulder"],
    }

    vectors = visuals.get("vectors", [])
    if not vectors:
        return visuals

    # Identify which body part is being corrected from the coaching script
    script_lower = coaching_script.lower()
    target_keypoints = []

    for body_part, kp_names in BODY_PART_KEYWORDS.items():
        if body_part in script_lower:
            target_keypoints.extend(kp_names)
            print(
                f"[Snap] Detected '{body_part}' in script, targeting keypoints: {kp_names}")

    # If no body part detected, fall back to finding closest keypoints
    if not target_keypoints:
        print(f"[Snap] No body part keywords found in script, using proximity matching")
        target_keypoints = list(KEYPOINTS.keys())

    # Separate skeleton and correction vectors
    correction_vectors = []
    skeleton_vectors = []

    for vec in vectors:
        color = vec.get("color", "").lower()
        if color in ["red", "#ff0000", "#ff4d4d", "#ff884d"] or "current" in str(vec.get("label", "")).lower():
            correction_vectors.append(vec)
        else:
            skeleton_vectors.append(vec)

    if not correction_vectors:
        return visuals

    print(f"[Snap] Processing {len(correction_vectors)} correction vectors...")

    # Map keypoints to their connected neighbors in the skeleton
    KEYPOINT_CONNECTIONS = {
        "left_shoulder": ["left_elbow", "left_hip"],
        "right_shoulder": ["right_elbow", "right_hip"],
        "left_elbow": ["left_shoulder", "left_wrist"],
        "right_elbow": ["right_shoulder", "right_wrist"],
        "left_wrist": ["left_elbow"],
        "right_wrist": ["right_elbow"],
        "left_hip": ["left_shoulder", "left_knee"],
        "right_hip": ["right_shoulder", "right_knee"],
        "left_knee": ["left_hip", "left_ankle"],
        "right_knee": ["right_hip", "right_ankle"],
        "left_ankle": ["left_knee"],
        "right_ankle": ["right_knee"],
    }

    snapped_vectors = []

    for cvec in correction_vectors:
        start = cvec.get("start", [0.5, 0.5])

        # Find the target keypoint that matches the body part mentioned in script
        best_kp_coords = None
        best_kp_name = None
        best_kp_idx = None
        min_dist = float('inf')

        for kp_name in target_keypoints:
            if kp_name in KEYPOINTS:
                kp_idx = KEYPOINTS[kp_name]
                kp_coords = get_keypoint_coords(
                    xy, conf, kp_idx, width, height)

                if kp_coords:
                    # Prioritize keypoints that match the coaching context
                    dist = ((start[0] - kp_coords[0])**2 +
                            (start[1] - kp_coords[1])**2)**0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_kp_coords = kp_coords
                        best_kp_name = kp_name
                        best_kp_idx = kp_idx

        if best_kp_coords and best_kp_name in KEYPOINT_CONNECTIONS:
            # Find an adjacent keypoint to form the angle
            adjacent_coords = None
            adjacent_name = None

            for adj_name in KEYPOINT_CONNECTIONS[best_kp_name]:
                if adj_name in KEYPOINTS:
                    adj_idx = KEYPOINTS[adj_name]
                    adj_coords = get_keypoint_coords(
                        xy, conf, adj_idx, width, height)
                    if adj_coords:
                        adjacent_coords = adj_coords
                        adjacent_name = adj_name
                        break

            if adjacent_coords:
                # Vector 1: Current limb segment (joint → adjacent keypoint) — RED
                snapped_vectors.append({
                    "start": list(best_kp_coords),
                    "end": list(adjacent_coords),
                    "color": "#FF3B30",
                    "label": "Current"
                })

                # Vector 2: Target position (joint → corrected position) — GREEN
                end = cvec.get("end", [0.5, 0.5])
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                # Get current segment direction and length
                curr_dx = adjacent_coords[0] - best_kp_coords[0]
                curr_dy = adjacent_coords[1] - best_kp_coords[1]
                segment_length = (curr_dx**2 + curr_dy**2)**0.5

                # Apply correction rotation (scale correction to match segment length)
                correction_length = (dx**2 + dy**2)**0.5
                if correction_length > 0:
                    scale = segment_length / correction_length
                    target_x = best_kp_coords[0] + dx * scale * 1.5
                    target_y = best_kp_coords[1] + dy * scale * 1.5
                else:
                    import math
                    angle = math.radians(30)
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    target_x = best_kp_coords[0] + \
                        (curr_dx * cos_a - curr_dy * sin_a)
                    target_y = best_kp_coords[1] + \
                        (curr_dx * sin_a + curr_dy * cos_a)

                snapped_vectors.append({
                    "start": list(best_kp_coords),
                    "end": [target_x, target_y],
                    "color": "#34C759",
                    "label": "Target"
                })

                print(
                    f"[Snap] ✓ Created angle pair at {best_kp_name}: {best_kp_name}→{adjacent_name} (current) vs target")
            else:
                # No adjacent keypoint found, create single correction vector
                end = cvec.get("end", [0.5, 0.5])
                dx = end[0] - start[0]
                dy = end[1] - start[1]

                snapped_vectors.append({
                    "start": list(best_kp_coords),
                    "end": [best_kp_coords[0] + dx, best_kp_coords[1] + dy],
                    "color": "#FF3B30",
                    "label": cvec.get("label")
                })
                print(
                    f"[Snap] ⚠ No adjacent keypoint for {best_kp_name}, created single vector")
        else:
            # Keep original if no keypoint found
            snapped_vectors.append(cvec)
            print(f"[Snap] ⚠ Could not snap correction, keeping original position")

    # Combine skeleton + snapped corrections
    visuals["vectors"] = skeleton_vectors + snapped_vectors

    return visuals

# ==============================================================================


video_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "git+https://github.com/huggingface/transformers.git",
        "accelerate",
        "qwen-vl-utils",
        "outlines",
        "pydantic",
        "decord",
        "torchvision",
        "huggingface_hub",
        "ultralytics",
        "opencv-python-headless",
    )
    # Bake model weights into the image at build time (runs once, cached forever)
    .run_function(
        download_model_weights,
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
)
tts_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl")
    .pip_install("kokoro_onnx", "soundfile", "misaki[en]", "numpy")
    # Pre-download weights so they are baked into the image
    .run_commands(
        "curl -L https://github.com/remsky/Kokoro-FastAPI/raw/main/model/kokoro-v1.0.onnx -o /root/kokoro-v1.0.onnx",
        "curl -L https://github.com/remsky/Kokoro-FastAPI/raw/main/model/voices-v1.0.bin -o /root/voices-v1.0.bin",
    )
)

app = modal.App("biomechanics-ai")


@app.cls(image=tts_image, gpu="T4", scaledown_window=60)
class TextToSpeech:
    @modal.enter()
    def setup(self):
        from kokoro_onnx import Kokoro
        from misaki import en

        # Load once and keep in memory
        self.model = Kokoro("/root/kokoro-v1.0.onnx", "/root/voices-v1.0.bin")
        self.g2p = en.G2P(trf=False, british=False)

    @modal.method()
    def speak(self, text: str) -> str:
        import soundfile as sf

        phonemes, _ = self.g2p(text)
        samples, sample_rate = self.model.create(
            phonemes, "af_heart", is_phonemes=True)

        wav_io = io.BytesIO()
        sf.write(wav_io, samples, sample_rate, format="WAV")
        return base64.b64encode(wav_io.getvalue()).decode("utf-8")


class Point(BaseModel):
    x: float
    y: float


class Vector(BaseModel):
    start: Tuple[float, float]
    end: Tuple[float, float]
    color: str
    label: Optional[str] = None


class VisualOverlay(BaseModel):
    overlay_type: Literal["POSITION_MARKER", "ANGLE_CORRECTION", "PATH_TRACE"]
    focus_point: Optional[Point] = None
    vectors: Optional[List[Vector]] = None
    path_points: Optional[List[Tuple[float, float]]] = None


class FeedbackPointLLM(BaseModel):
    mistake_timestamp_ms: int
    coaching_script: str
    visuals: Optional[VisualOverlay] = None


class FeedbackPointResponse(FeedbackPointLLM):
    audio_url: str


class BiomechanicalAnalysisLLM(BaseModel):
    status: Literal["success", "low_confidence", "error"]
    error_message: Optional[str] = None
    positive_note: str
    progress_score: int
    improvement_delta: Optional[int] = None
    feedback_points: List[FeedbackPointLLM]


class BiomechanicalAnalysisResponse(BiomechanicalAnalysisLLM):
    feedback_points: List[FeedbackPointResponse]


@app.cls(
    gpu="B200",
    image=video_image,
    timeout=600,  # Max 10 mins per analysis
    ephemeral_disk=512 * 1024,  # Minimum 512 GiB required by Modal for large models
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class VideoAnalyzer:
    @modal.enter()
    def load_model(self):
        # Weights are baked into the image at MODEL_CACHE_DIR — no download at runtime
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        self.raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_CACHE_DIR, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE_DIR)

    @modal.method()
    def analyze(self, video_bytes: bytes, user_description: str, activity_type: str, previous_analysis: str = "", pose_data: Optional[dict] = None):
        import json
        import time as _time

        print(f"\n{'='*60}")
        print(f"[Analyze] 🎬 Starting VL analysis")
        print(f"[Analyze]   activity_type = {activity_type}")
        print(f"[Analyze]   user_description = {user_description}")
        print(f"[Analyze]   video_bytes = {len(video_bytes):,} bytes")
        print(f"[Analyze]   has_previous_analysis = {bool(previous_analysis)}")
        print(
            f"[Analyze]   has_pose_data = {pose_data is not None and bool(pose_data.get('detected_poses'))}")
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
            activity_type, "Analyze the user's body mechanics, posture, and movement pattern for this activity.")

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            # ── Build the coaching prompt ──
            is_retry = bool(previous_analysis)
            prompt = f"""You are MotionCoach, a supportive and encouraging AI movement coach.

Your personality:
- Warm, positive, and motivating — like a friendly personal trainer, NOT a drill sergeant
- Always lead with what the user is doing WELL before suggesting corrections
- Use concise, actionable language a beginner can understand
- Reference specific body parts and timestamps so feedback is precise

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
- Compute improvement_delta: positive number = they improved, negative = they regressed, 0 = no change
- If they fixed a previous mistake, acknowledge it enthusiastically (e.g. "Great job fixing your elbow angle!")
- Do NOT repeat feedback for issues they already corrected
- Focus new feedback on the NEXT most impactful improvement they should make
- Adjust progress_score relative to their previous score
"""
            else:
                prompt += """\n=== NEW SESSION — FIRST ATTEMPT ===
This is the user's first attempt at this movement. There is no previous session to compare to.
- Set improvement_delta to null (no previous baseline)
- Give an honest baseline progress_score (0-100)
- Be encouraging: this is their starting point, not a judgment
"""

            # Add detected pose skeleton information
            if pose_data and pose_data.get("detected_poses"):
                prompt += "\n=== DETECTED POSE KEYPOINTS (from YOLO26x-Pose) ===\n"
                prompt += "These are ground-truth body keypoint coordinates detected by computer vision.\n"
                prompt += "All coordinates are normalized floats [0.0, 1.0] where (0,0)=top-left, (1,1)=bottom-right.\n\n"
                for timestamp_ms, keypoints_data in pose_data["detected_poses"].items():
                    prompt += f"Frame at {timestamp_ms}ms ({int(timestamp_ms)//1000}s): {keypoints_data['num_people']} person(s)\n"
                    if keypoints_data.get('keypoints'):
                        for kp_name, coords in keypoints_data['keypoints'].items():
                            prompt += f"  {kp_name}: [{coords[0]:.3f}, {coords[1]:.3f}]\n"
                prompt += """\nRULES FOR VISUAL OVERLAYS:
- The full body skeleton is drawn AUTOMATICALLY by the backend — do NOT include skeleton vectors in your output
- You ONLY output correction vectors showing what needs to change
- Always specify LEFT or RIGHT in your coaching_script (e.g. "left elbow" not just "elbow")
- For "Current" vectors: use the detected keypoint as the start, and the current wrong neighbor as the end
- For "Target" vectors: use the same start keypoint, and show where it SHOULD go
"""

            prompt += """\n=== OUTPUT RULES ===
1. Return between 1 and 5 feedback_points, ordered by importance
2. Each coaching_script should be 1-2 sentences. ALWAYS specify LEFT or RIGHT when referring to a body part
3. The positive_note should genuinely highlight something the user did well
4. progress_score is 0-100 rating of overall form quality
5. All visual coordinates MUST be floats between 0.0 and 1.0
6. For ANGLE_CORRECTION visuals: include exactly 2 vectors:
   - One labeled "Current" with color "red" — the current wrong limb position
   - One labeled "Target" with color "green" — the corrected limb position
   - Both must start from the SAME joint keypoint
7. For POSITION_MARKER visuals: set focus_point to the relevant body part coordinate, no vectors needed
8. Output ONLY valid JSON — no markdown fences, no extra text
9. Do NOT draw the full skeleton — the backend does that. Only draw correction arrows.
"""

            prompt += """
=== EXAMPLE OUTPUT FORMAT ===
Your response must be a single JSON object with EXACTLY this structure (fill in real values):
{
  "status": "success",
  "error_message": null,
  "positive_note": "Your stance looks solid and balanced — great foundation!",
  "progress_score": 55,
  "improvement_delta": null,
  "feedback_points": [
    {
      "mistake_timestamp_ms": 2000,
      "coaching_script": "At 2 seconds, try raising your LEFT elbow higher during the backswing for more power.",
      "visuals": {
        "overlay_type": "ANGLE_CORRECTION",
        "focus_point": {"x": 0.45, "y": 0.55},
        "vectors": [
          {"start": [0.45, 0.55], "end": [0.50, 0.65], "color": "red", "label": "Current"},
          {"start": [0.45, 0.55], "end": [0.40, 0.40], "color": "green", "label": "Target"}
        ],
        "path_points": null
      }
    }
  ]
}

IMPORTANT: Do NOT output the schema definition. Output actual coaching feedback as a JSON object.
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

            # Sample up to 24 fps for the first 5 seconds
            end_time = min(5.0, total_duration)
            nframes = max(1, int(end_time * 24.0))
            frame_indices = np.linspace(
                0, min(int(end_time * actual_fps), total_frames) - 1,
                num=nframes, dtype=int
            ).tolist()
            raw_frames = vr.get_batch(frame_indices).asnumpy()
            video_frames = [PILImage.fromarray(f) for f in raw_frames]
            del vr

            print(
                f"[Analyze] 🎞️  Extracted {len(video_frames)} frames from {total_duration:.1f}s video ({actual_fps:.1f} native fps, sampled {nframes} frames over {end_time:.1f}s)")

            # Build messages with pre-decoded PIL frames
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_frames},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # === Official Qwen3-VL pipeline (one-step apply_chat_template) ===
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")

            # Log model input tensor shapes
            print(f"[Analyze] 🧠 Model input tensors:")
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

            # Fix: Qwen3-VL creates per-frame vision blocks, each needing its own
            # grid_thw entry. Expand [1, H, W] → [num_frames, H, W] if mismatched.
            import itertools
            import torch
            mm_types = inputs["mm_token_type_ids"][0].tolist()
            type_groups = [(k, len(list(g)))
                           for k, g in itertools.groupby(mm_types)]
            video_groups = [(k, l) for k, l in type_groups if k == 2]
            n_video_groups = len(video_groups)
            n_grid_entries = inputs["video_grid_thw"].shape[0]
            if n_video_groups > n_grid_entries:
                orig_thw = inputs["video_grid_thw"][0]
                t_val, h_val, w_val = orig_thw[0].item(
                ), orig_thw[1].item(), orig_thw[2].item()
                expanded = torch.tensor(
                    [[1, h_val, w_val]] * n_video_groups,
                    dtype=orig_thw.dtype, device=orig_thw.device
                )
                inputs["video_grid_thw"] = expanded
                print(
                    f"[Analyze] ⚠️  Expanded video_grid_thw: {n_grid_entries} → {n_video_groups} entries")

            # Native Hugging Face generation
            t_gen_start = _time.time()
            print(f"[Analyze] 🚀 Starting model.generate (max_new_tokens=4096)...")
            generated_ids = self.raw_model.generate(
                **inputs, max_new_tokens=4096)
            t_gen_end = _time.time()
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            print(
                f"[Analyze] ✅ Generation complete in {t_gen_end - t_gen_start:.1f}s")
            print(
                f"[Analyze] --- RAW MODEL OUTPUT ({len(output_text)} chars) ---")
            print(output_text[:3000])
            if len(output_text) > 3000:
                print(f"  ... ({len(output_text) - 3000} more chars) ...")
            print(f"[Analyze] --- RAW MODEL OUTPUT END ---")

            # Clean markdown code blocks if the LLM added them
            import re
            json_match = re.search(
                r'```json\n*(.*?)\n*```', output_text, re.DOTALL)
            clean_json_str = json_match.group(
                1) if json_match else output_text.strip()

            # Validate and dump
            result = BiomechanicalAnalysisLLM.model_validate_json(
                clean_json_str).model_dump()

            # Cap feedback points at 5
            if len(result.get("feedback_points", [])) > 5:
                print(
                    f"[Analyze] ✂️  Capping feedback_points from {len(result['feedback_points'])} to 5")
                result["feedback_points"] = result["feedback_points"][:5]

            t_end = _time.time()
            print(f"\n[Analyze] 📊 Parsed result summary:")
            print(f"  status: {result.get('status')}")
            print(f"  progress_score: {result.get('progress_score')}")
            print(f"  improvement_delta: {result.get('improvement_delta')}")
            print(f"  positive_note: {result.get('positive_note', '')[:100]}")
            print(
                f"  feedback_points: {len(result.get('feedback_points', []))}")
            for i, fp in enumerate(result.get('feedback_points', [])):
                print(
                    f"    [{i}] @{fp.get('mistake_timestamp_ms')}ms: {fp.get('coaching_script', '')[:80]}...")
            print(
                f"[Analyze] ⏱️  Total analyze time: {t_end - t_start:.1f}s (generation: {t_gen_end - t_gen_start:.1f}s)")
            print(f"{'='*60}\n")

        return result

    @modal.method()
    def extract_pose_data(self, video_bytes: bytes, timestamps_ms: List[int]) -> dict:
        """Extract pose keypoints at specific timestamps for LLM context."""
        import cv2
        from ultralytics import YOLO

        print(
            f"[Pose] Extracting pose data at {len(timestamps_ms)} timestamps...")
        model = YOLO("yolo26x-pose.pt")

        # COCO keypoint names for clarity
        KEYPOINT_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
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

            for ts in timestamps_ms:
                cap.set(cv2.CAP_PROP_POS_MSEC, ts)
                success, frame = cap.read()

                if not success or frame is None:
                    continue

                results = model(frame, verbose=False, conf=0.15)

                if not results or len(results) == 0:
                    continue

                result = results[0]
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
                                round(y / height, 3)
                            ]

                    detected_poses[str(ts)] = {
                        "num_people": len(xy),
                        "keypoints": keypoints_dict
                    }
                    print(
                        f"[Pose] ✓ Extracted {len(keypoints_dict)} keypoints at {ts}ms")

            cap.release()

        return {"detected_poses": detected_poses}

    @modal.method()
    def add_pose_overlays(self, video_bytes: bytes, feedback_points: List[dict]) -> List[dict]:
        """Extract pose skeleton for each feedback timestamp and merge with visuals."""
        import cv2
        import numpy as np
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
                f"[Pose] Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

            for fp_idx, fp in enumerate(feedback_points):
                timestamp_ms = fp.get("mistake_timestamp_ms", 0)
                visuals = fp.get("visuals")
                coaching_script = fp.get("coaching_script", "")

                # If no visuals exist, create a basic structure for the skeleton
                if not visuals:
                    print(
                        f"[Pose] Creating empty visuals for feedback point {fp_idx}")
                    visuals = {
                        "overlay_type": "POSITION_MARKER",
                        "focus_point": None,
                        "vectors": []
                    }
                    fp["visuals"] = visuals

                if not timestamp_ms:
                    print(
                        f"[Pose] Skipping feedback point {fp_idx}: no timestamp")
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
                        results = model(frame, verbose=False,
                                        conf=0.15)  # Lower confidence

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
                            xy, conf, width, height, min_conf=0.15)

                        if pose_vectors:
                            print(
                                f"[Pose] ✓ Found {len(pose_vectors)} skeleton vectors at {ts}ms for feedback {fp_idx}")

                            # Snap any existing correction vectors to detected skeleton for better alignment
                            if visuals.get("vectors"):
                                print(
                                    f"[Pose] Snapping {len(visuals['vectors'])} correction vectors to skeleton...")
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
                                "elbow":    {"left": [7], "right": [8], "both": [7, 8]},
                                "arm":      {"left": [5, 7, 9], "right": [6, 8, 10], "both": [5, 6, 7, 8, 9, 10]},
                                "wrist":    {"left": [9], "right": [10], "both": [9, 10]},
                                "hand":     {"left": [9], "right": [10], "both": [9, 10]},
                                "hip":      {"left": [11], "right": [12], "both": [11, 12]},
                                "knee":     {"left": [13], "right": [14], "both": [13, 14]},
                                "leg":      {"left": [11, 13, 15], "right": [12, 14, 16], "both": [11, 12, 13, 14, 15, 16]},
                                "ankle":    {"left": [15], "right": [16], "both": [15, 16]},
                                "foot":     {"left": [15], "right": [16], "both": [15, 16]},
                                "back":     {"left": [5, 6, 11, 12], "right": [5, 6, 11, 12], "both": [5, 6, 11, 12]},
                                "spine":    {"left": [5, 6, 11, 12], "right": [5, 6, 11, 12], "both": [5, 6, 11, 12]},
                                "head":     {"left": [0, 3, 4], "right": [0, 3, 4], "both": [0, 3, 4]},
                                "neck":     {"left": [0, 5, 6], "right": [0, 5, 6], "both": [0, 5, 6]},
                            }
                            for kw, sides in BODY_PART_KW.items():
                                if kw in script_lower:
                                    # Pick the right side based on what the script mentions
                                    if has_left and not has_right:
                                        critiqued_kp_indices.update(
                                            sides["left"])
                                    elif has_right and not has_left:
                                        critiqued_kp_indices.update(
                                            sides["right"])
                                    else:
                                        critiqued_kp_indices.update(
                                            sides["both"])

                            if critiqued_kp_indices:
                                print(
                                    f"[Pose] 🎯 Critiqued keypoint indices: {sorted(critiqued_kp_indices)}")
                                # Build a set of (kp_a, kp_b) pairs that should be red
                                # BOTH endpoints must be in the critiqued set to avoid color bleed
                                critiqued_segments = set()
                                for a, b in COCO_SKELETON:
                                    i_a, i_b = a - 1, b - 1  # 0-indexed
                                    if i_a in critiqued_kp_indices and i_b in critiqued_kp_indices:
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
                                        sx, sy = float(
                                            person[i_a][0] / width), float(person[i_a][1] / height)
                                        ex, ey = float(
                                            person[i_b][0] / width), float(person[i_b][1] / height)
                                        if (abs(vs[0]-sx) < 0.01 and abs(vs[1]-sy) < 0.01 and
                                                abs(ve[0]-ex) < 0.01 and abs(ve[1]-ey) < 0.01):
                                            vec["color"] = "#FF3B30"
                                            recolored += 1
                                            break
                                print(
                                    f"[Pose] 🎨 Recolored {recolored}/{len(critiqued_segments)} critiqued skeleton segments to red")

                            # Merge skeleton with (now-snapped) correction vectors
                            existing_vectors = visuals.get("vectors") or []
                            combined_vectors = pose_vectors + existing_vectors
                            visuals["vectors"] = combined_vectors

                            print(
                                f"[Pose] Combined {len(pose_vectors)} skeleton + {len(existing_vectors)} correction vectors")
                            break
                        else:
                            print(f"[Pose] No valid keypoints at {ts}ms")

                    if not pose_vectors:
                        print(
                            f"[Pose] ✗ No pose detected for feedback {fp_idx} at {timestamp_ms}ms")
                        continue

                except Exception as e:
                    print(f"[Pose] ERROR at {timestamp_ms}ms: {e}")
                    import traceback
                    traceback.print_exc()

            cap.release()

        return feedback_points


@app.function(image=modal.Image.debian_slim().pip_install("fastapi", "python-multipart"), timeout=600)
@modal.fastapi_endpoint(method="POST")
async def analyze(request: Request):
    import json
    import asyncio
    import time as _time

    t_request_start = _time.time()

    # 1. Parse Multipart Form Data
    form = await request.form()

    video_file = form.get("video_file")
    activity_type = form.get("activity_type", "Unknown Activity")
    user_description = form.get("user_description", "")
    previous_analysis_str = form.get("previous_analysis", "")

    print(f"\n{'#'*60}")
    print(f"[Endpoint] 📥 Incoming POST /analyze")
    print(f"[Endpoint]   activity_type = {activity_type}")
    print(f"[Endpoint]   user_description = {user_description[:100]}")
    print(
        f"[Endpoint]   has_previous_analysis = {bool(previous_analysis_str)}")
    print(f"[Endpoint]   has_video_file = {video_file is not None}")

    if not video_file:
        print(f"[Endpoint] ❌ No video_file in form data")
        return Response(content=json.dumps({
            "status": "error",
            "error_message": "Missing video_file in form data.",
            "positive_note": "",
            "progress_score": 0,
            "feedback_points": []
        }), media_type="application/json")

    video_bytes = await video_file.read()
    print(
        f"[Endpoint]   video_size = {len(video_bytes):,} bytes ({len(video_bytes)/1024/1024:.1f} MB)")

    # 2. Extract pose data from video for LLM context
    # Sample frames at 1s, 2s, 3s, 4s, 5s
    sample_timestamps = [1000, 2000, 3000, 4000, 5000]
    try:
        t_pose_start = _time.time()
        print(
            f"[Endpoint] 🦴 Extracting pose data at timestamps: {sample_timestamps}...")
        pose_data = await VideoAnalyzer().extract_pose_data.remote.aio(
            video_bytes=video_bytes,
            timestamps_ms=sample_timestamps
        )
        t_pose_end = _time.time()
        n_poses = len(pose_data.get('detected_poses', {}))
        print(
            f"[Endpoint] ✅ Pose data extracted: {n_poses} frames in {t_pose_end - t_pose_start:.1f}s")
    except Exception as e:
        print(
            f"[Endpoint] ⚠️  Pose extraction failed, continuing without pose context: {e}")
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
            pose_data=pose_data
        )
        t_llm_end = _time.time()
        print(
            f"[Endpoint] ✅ VL analysis complete in {t_llm_end - t_llm_start:.1f}s")
        print(
            f"[Endpoint]   LLM status={llm_response.get('status')}, score={llm_response.get('progress_score')}, feedback_points={len(llm_response.get('feedback_points', []))}")
    except Exception as e:
        print(f"[Endpoint] ❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return Response(content=json.dumps({
            "status": "error",
            "error_message": f"Analysis failed: {str(e)}",
            "positive_note": "",
            "progress_score": 0,
            "feedback_points": []
        }), media_type="application/json")

    if llm_response.get("status") == "error":
        print(
            f"[Endpoint] ❌ Model returned error: {llm_response.get('error_message')}")
        return Response(content=json.dumps(llm_response), media_type="application/json")

    # 4. Generate TTS for all feedback points concurrently
    feedback_points = llm_response.get("feedback_points", [])
    audio_results = [""] * len(feedback_points)

    # 5. Add pose skeleton overlays to each feedback timestamp
    try:
        t_overlay_start = _time.time()
        print(
            f"[Endpoint] 🦴 Adding pose skeleton overlays to {len(feedback_points)} feedback points...")
        feedback_points = await VideoAnalyzer().add_pose_overlays.remote.aio(
            video_bytes=video_bytes,
            feedback_points=feedback_points
        )
        t_overlay_end = _time.time()
        print(
            f"[Endpoint] ✅ Pose skeleton overlays added in {t_overlay_end - t_overlay_start:.1f}s")
    except Exception as e:
        print(f"[Endpoint] ⚠️  Pose overlay generation failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Construct Final Response
    final_feedback_points = []
    for fp, audio_b64 in zip(feedback_points, audio_results):
        fp_copy = fp.copy()
        fp_copy["audio_url"] = f"data:audio/wav;base64,{audio_b64}"
        final_feedback_points.append(fp_copy)

    llm_response["feedback_points"] = final_feedback_points

    # Log the final JSON payload (truncate audio to avoid log spam)
    t_request_end = _time.time()
    print(f"\n[Endpoint] 📤 FINAL RESPONSE PAYLOAD:")
    log_response = json.loads(json.dumps(llm_response))  # deep copy
    for fp in log_response.get("feedback_points", []):
        if "audio_url" in fp:
            fp["audio_url"] = fp["audio_url"][:50] + "...[TRUNCATED]"
        # Truncate vectors list for readability
        if fp.get("visuals") and fp["visuals"].get("vectors"):
            n_vecs = len(fp["visuals"]["vectors"])
            if n_vecs > 3:
                fp["visuals"]["vectors"] = fp["visuals"]["vectors"][:3]
                fp["visuals"]["vectors"].append(
                    {"_truncated": f"...and {n_vecs - 3} more vectors"})
    print(json.dumps(log_response, indent=2))
    print(
        f"\n[Endpoint] ⏱️  Total request time: {t_request_end - t_request_start:.1f}s")
    print(f"{'#'*60}\n")

    return Response(content=json.dumps(llm_response), media_type="application/json")
