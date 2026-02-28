"""
# ==============================================================================
# 🦴 YOLO POSE DETECTION FOR MOTION COACH
# ==============================================================================
# Purpose:
#   Extract skeleton keypoints from video frames using YOLO pose estimation.
#   Convert COCO keypoints to the app's vector format for SVGOverlay rendering.
#
# Integration:
#   Called by VideoAnalyzer to add skeleton overlays to coaching feedback.
# ==============================================================================
"""
from typing import List, Tuple, Optional
import modal

# COCO skeleton connections (17 keypoints total)
COCO_SKELETON = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),  # Face/head
    (6, 12), (7, 13), (6, 7),  # Shoulders to hips
    (6, 8), (7, 9), (8, 10), (9, 11),  # Arms
    (2, 3), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7),  # Torso/legs
]

pose_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        "ultralytics",
        "opencv-python-headless",
        "numpy",
    )
)


def keypoints_to_vectors(
    xy: list,
    conf: list,
    width: int,
    height: int,
    min_conf: float = 0.25,
    color: str = "#00FFFF",  # Cyan skeleton
) -> List[dict]:
    """
    Convert YOLO keypoints to app's vector format.
    
    Args:
        xy: Keypoint coordinates (17 points per person)
        conf: Confidence scores per keypoint
        width: Video frame width
        height: Video frame height
        min_conf: Minimum confidence threshold
        color: Skeleton line color
    
    Returns:
        List of vector dicts with normalized coordinates
    """
    vectors = []
    
    for person_idx, person in enumerate(xy):
        # Draw skeleton connections
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


def extract_keypoints_from_frame(
    frame_bytes: bytes,
    model_path: str = "yolo26n-pose.pt",
    min_conf: float = 0.25,
) -> Tuple[Optional[list], Optional[list], int, int]:
    """
    Run YOLO pose detection on a single frame.
    
    Returns:
        (xy_coords, confidences, width, height)
    """
    import cv2
    import numpy as np
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(model_path)
    
    # Decode frame
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return None, None, 0, 0
    
    height, width = frame.shape[:2]
    
    # Run inference
    results = model(frame, verbose=False)
    
    if not results or len(results) == 0:
        return None, None, width, height
    
    result = results[0]
    keypoints = result.keypoints
    
    if keypoints is None:
        return None, None, width, height
    
    xy = keypoints.xy.cpu().numpy()
    conf = keypoints.conf
    conf = conf.cpu().numpy() if conf is not None else None
    
    return xy.tolist(), conf.tolist() if conf is not None else None, width, height


def extract_pose_at_timestamp(
    video_bytes: bytes,
    timestamp_ms: int,
    model_path: str = "yolo11n-pose.pt",
) -> List[dict]:
    """
    Extract pose skeleton at a specific video timestamp.
    
    Args:
        video_bytes: Raw video file bytes
        timestamp_ms: Millisecond to extract pose from
        model_path: YOLO model to use
    
    Returns:
        List of vectors for SVGOverlay
    """
    import cv2
    import tempfile
    import numpy as np
    from ultralytics import YOLO
    
    # Write video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp.flush()
        
        cap = cv2.VideoCapture(tmp.name)
        
        if not cap.isOpened():
            return []
        
        # Seek to timestamp
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        
        success, frame = cap.read()
        cap.release()
        
        if not success or frame is None:
            return []
        
        height, width = frame.shape[:2]
        
        # Run pose detection
        model = YOLO(model_path)
        results = model(frame, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        result = results[0]
        keypoints = result.keypoints
        
        if keypoints is None:
            return []
        
        xy = keypoints.xy.cpu().numpy().tolist()
        conf = keypoints.conf
        conf = conf.cpu().numpy().tolist() if conf is not None else None
        
        # Convert to app vector format
        return keypoints_to_vectors(xy, conf, width, height)
