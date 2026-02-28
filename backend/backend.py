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
    color: str = "#00FFFF",  # Cyan skeleton
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
            print(f"[Snap] Detected '{body_part}' in script, targeting keypoints: {kp_names}")
    
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
                kp_coords = get_keypoint_coords(xy, conf, kp_idx, width, height)
                
                if kp_coords:
                    # Prioritize keypoints that match the coaching context
                    dist = ((start[0] - kp_coords[0])**2 + (start[1] - kp_coords[1])**2)**0.5
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
                    adj_coords = get_keypoint_coords(xy, conf, adj_idx, width, height)
                    if adj_coords:
                        adjacent_coords = adj_coords
                        adjacent_name = adj_name
                        break
            
            if adjacent_coords:
                # Vector 1: Current limb segment (joint → adjacent keypoint)
                snapped_vectors.append({
                    "start": list(best_kp_coords),
                    "end": list(adjacent_coords),
                    "color": cvec.get("color", "#FF3B30"),
                    "label": "Current"
                })
                
                # Vector 2: Target position (joint → corrected position)
                # Use the correction direction to calculate target endpoint
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
                    # Normalize correction and scale to segment length
                    scale = segment_length / correction_length
                    target_x = best_kp_coords[0] + dx * scale * 1.5
                    target_y = best_kp_coords[1] + dy * scale * 1.5
                else:
                    # Fallback: rotate current segment by 30 degrees
                    import math
                    angle = math.radians(30)
                    cos_a = math.cos(angle)
                    sin_a = math.sin(angle)
                    target_x = best_kp_coords[0] + (curr_dx * cos_a - curr_dy * sin_a)
                    target_y = best_kp_coords[1] + (curr_dx * sin_a + curr_dy * cos_a)
                
                snapped_vectors.append({
                    "start": list(best_kp_coords),
                    "end": [target_x, target_y],
                    "color": cvec.get("color", "#FF3B30"),
                    "label": "Target"
                })
                
                print(f"[Snap] ✓ Created angle pair at {best_kp_name}: {best_kp_name}→{adjacent_name} (current) vs target")
            else:
                # No adjacent keypoint found, create single correction vector
                end = cvec.get("end", [0.5, 0.5])
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                
                snapped_vectors.append({
                    "start": list(best_kp_coords),
                    "end": [best_kp_coords[0] + dx, best_kp_coords[1] + dy],
                    "color": cvec.get("color", "red"),
                    "label": cvec.get("label")
                })
                print(f"[Snap] ⚠ No adjacent keypoint for {best_kp_name}, created single vector")
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
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .run_commands(
        "pip install --upgrade pip",
        "pip install torch --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers>=4.48.2",
        "accelerate",
        "qwen-vl-utils",
        "outlines",
        "pydantic",
        "decord",
        "torchvision",
        "ultralytics",
        "opencv-python-headless",
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
    # Choose GPU based on model size (A10G is usually enough for 7B)
    gpu="B200",
    image=video_image,
    timeout=600,  # Max 10 mins per analysis
)
class VideoAnalyzer:
    @modal.enter()
    def load_model(self):
        # This only runs once when the container starts
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

    @modal.method()
    def analyze(self, video_bytes: bytes, user_description: str, activity_type: str, previous_analysis: str = "", pose_data: Optional[dict] = None):
        from qwen_vl_utils import process_vision_info
        import json

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            prompt = f"Act as a professional coach for {activity_type}. The user's goal is: {user_description}. "
            if previous_analysis:
                prompt += f"\n\nHere is the context from the user's PREVIOUS attempt:\n{previous_analysis}\n\nUse this to compute the improvement_delta and avoid repeating the exact same feedback if they improved."

            # Add detected pose skeleton information
            if pose_data and pose_data.get("detected_poses"):
                prompt += "\n\n=== DETECTED POSE DATA ===\n"
                prompt += "Below are the actual detected body keypoints from pose estimation. Use these EXACT coordinates when creating visual overlays.\n\n"
                for timestamp_ms, keypoints_data in pose_data["detected_poses"].items():
                    prompt += f"\nAt {timestamp_ms}ms:\n"
                    prompt += f"- Detected {keypoints_data['num_people']} person(s)\n"
                    if keypoints_data.get('keypoints'):
                        prompt += "- Body keypoints (normalized 0-1 coordinates):\n"
                        for kp_name, coords in keypoints_data['keypoints'].items():
                            prompt += f"  • {kp_name}: [{coords[0]:.3f}, {coords[1]:.3f}]\n"
                prompt += "\n⚠️ IMPORTANT: When creating correction visuals:\n"
                prompt += "1. Use ONLY the detected keypoint coordinates above\n"
                prompt += "2. If correcting 'elbow', use left_elbow or right_elbow coordinates as the vector start point\n"
                prompt += "3. If correcting 'back', use shoulder and hip keypoints\n"
                prompt += "4. If correcting 'knee', use left_knee or right_knee coordinates\n"
                prompt += "5. Match the body part mentioned in your coaching_script to the corresponding keypoint\n"

            prompt += "\n\nProvide coaching feedback. All coordinates for visuals MUST be relative floats between 0.0 and 1.0 (e.g. [0.4, 0.5]), where [0.0, 0.0] is top-left and [1.0, 1.0] is bottom-right."

            schema_json = BiomechanicalAnalysisLLM.model_json_schema()
            prompt += f"\n\nOutput ONLY a valid JSON object matching this JSON Schema:\n{json.dumps(schema_json)}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": tmp.name, "fps": 2.0},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # Native Hugging Face generation
            generated_ids = self.raw_model.generate(
                **inputs, max_new_tokens=4096)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # Clean markdown code blocks if the LLM added them
            import re
            json_match = re.search(
                r'```json\n*(.*?)\n*```', output_text, re.DOTALL)
            clean_json_str = json_match.group(
                1) if json_match else output_text.strip()

            # Validate and dump
            result = BiomechanicalAnalysisLLM.model_validate_json(
                clean_json_str).model_dump()
        return result

    @modal.method()
    def extract_pose_data(self, video_bytes: bytes, timestamps_ms: List[int]) -> dict:
        """Extract pose keypoints at specific timestamps for LLM context."""
        import cv2
        from ultralytics import YOLO
        
        print(f"[Pose] Extracting pose data at {len(timestamps_ms)} timestamps...")
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
                    print(f"[Pose] ✓ Extracted {len(keypoints_dict)} keypoints at {ts}ms")
            
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
            
            print(f"[Pose] Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")
            
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
                        "vectors": []
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
                        results = model(frame, verbose=False, conf=0.15)  # Lower confidence
                        
                        if not results or len(results) == 0:
                            continue
                        
                        result = results[0]
                        keypoints = result.keypoints
                        
                        if keypoints is None or keypoints.xy is None:
                            continue
                        
                        xy = keypoints.xy.cpu().numpy().tolist()
                        conf = keypoints.conf
                        conf = conf.cpu().numpy().tolist() if conf is not None else None
                        
                        # Convert to app vector format
                        pose_vectors = keypoints_to_vectors(xy, conf, width, height, min_conf=0.15)
                        
                        if pose_vectors:
                            print(f"[Pose] ✓ Found {len(pose_vectors)} skeleton vectors at {ts}ms for feedback {fp_idx}")
                            
                            # Snap any existing correction vectors to detected skeleton for better alignment
                            if visuals.get("vectors"):
                                print(f"[Pose] Snapping {len(visuals['vectors'])} correction vectors to skeleton...")
                                visuals = snap_correction_vectors_to_skeleton(
                                    visuals, xy, conf, width, height, coaching_script
                                )
                            
                            # Merge skeleton with (now-snapped) correction vectors
                            existing_vectors = visuals.get("vectors", [])
                            combined_vectors = pose_vectors + existing_vectors
                            visuals["vectors"] = combined_vectors
                            
                            print(f"[Pose] Combined {len(pose_vectors)} skeleton + {len(existing_vectors)} correction vectors")
                            break
                        else:
                            print(f"[Pose] No valid keypoints at {ts}ms")
                    
                    if not pose_vectors:
                        print(f"[Pose] ✗ No pose detected for feedback {fp_idx} at {timestamp_ms}ms")
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

    # 1. Parse Multipart Form Data
    # FastAPI request.form() handles multipart parsing
    form = await request.form()

    video_file = form.get("video_file")
    activity_type = form.get("activity_type", "Unknown Activity")
    user_description = form.get("user_description", "")
    previous_analysis_str = form.get("previous_analysis", "")

    if not video_file:
        return Response(content=json.dumps({
            "status": "error",
            "error_message": "Missing video_file in form data.",
            "positive_note": "",
            "progress_score": 0,
            "feedback_points": []
        }), media_type="application/json")

    video_bytes = await video_file.read()

    # 2. Extract pose data from video for LLM context
    # Sample frames at 1s, 2s, 3s, 4s, 5s
    sample_timestamps = [1000, 2000, 3000, 4000, 5000]
    try:
        print(f"[Endpoint] Extracting pose data at timestamps: {sample_timestamps}...")
        pose_data = await VideoAnalyzer().extract_pose_data.remote.aio(
            video_bytes=video_bytes,
            timestamps_ms=sample_timestamps
        )
        print(f"[Endpoint] ✓ Pose data extracted: {len(pose_data.get('detected_poses', {}))} frames")
    except Exception as e:
        print(f"[Endpoint] ⚠ Pose extraction failed, continuing without pose context: {e}")
        pose_data = None

    # 3. Call the VL model with pose context
    try:
        # We need to await the remote call if we make the router async
        llm_response = await VideoAnalyzer().analyze.remote.aio(
            video_bytes=video_bytes,
            user_description=user_description,
            activity_type=activity_type,
            previous_analysis=previous_analysis_str,
            pose_data=pose_data
        )
    except Exception as e:
        return Response(content=json.dumps({
            "status": "error",
            "error_message": f"Analysis failed: {str(e)}",
            "positive_note": "",
            "progress_score": 0,
            "feedback_points": []
        }), media_type="application/json")

    if llm_response.get("status") == "error":
        # Model explicitly failed to analyze
        return Response(content=json.dumps(llm_response), media_type="application/json")

    # 4. Generate TTS for all feedback points concurrently
    # tts_engine = TextToSpeech()
    feedback_points = llm_response.get("feedback_points", [])

    # We will gather all the remote calls and wait for them
    # Note: Modal .remote() is synchronous in standard python, so we map over them using .map

    # Prepare the scripts
    # scripts = [fp["coaching_script"] for fp in feedback_points]

    # Run TTS in parallel
    # if scripts:
    #     audio_results = []
    #     async for res in tts_engine.speak.map.aio(scripts):
    #         audio_results.append(res)
    # else:
    #     audio_results = []
    audio_results = [""] * len(feedback_points)

    # 5. Add pose skeleton overlays to each feedback timestamp
    # Run pose detection in VideoAnalyzer where opencv is available
    try:
        print(f"[Endpoint] Adding pose skeleton overlays to {len(feedback_points)} feedback points...")
        feedback_points = await VideoAnalyzer().add_pose_overlays.remote.aio(
            video_bytes=video_bytes,
            feedback_points=feedback_points
        )
        print(f"[Endpoint] ✓ Pose skeleton overlays added")
    except Exception as e:
        print(f"[Endpoint] ⚠ Pose overlay generation failed: {e}")
        import traceback
        traceback.print_exc()

    # 6. Construct Final Response
    final_feedback_points = []
    for fp, audio_b64 in zip(feedback_points, audio_results):
        fp_copy = fp.copy()
        fp_copy["audio_url"] = f"data:audio/wav;base64,{audio_b64}"
        final_feedback_points.append(fp_copy)

    llm_response["feedback_points"] = final_feedback_points

    return Response(content=json.dumps(llm_response), media_type="application/json")
