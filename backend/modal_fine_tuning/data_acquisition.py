"""
data_acquisition.py — Distributed dataset generation on Modal
=============================================================
Uses YOLO to filter images to exactly 1 person, then Gemini to generate
structured coaching JSON labels (distillation) for MotionCoach.

Prerequisites (run once):
    modal secret create gemini-api-key GEMINI_API_KEY=<your-key>
    modal secret create kaggle-secret KAGGLE_API_TOKEN=<your-token>

Usage:
    cd backend/modal_fine_tuning
    uv run modal run data_acquisition.py
    uv run modal run data_acquisition.py --max-samples 1000
"""

import os
import json
import modal

# ---------------------------------------------------------------------------
# Modal infrastructure
# ---------------------------------------------------------------------------
app = modal.App("qwen3-vl-distill-dataset")
vol = modal.Volume.from_name("qwen-finetune-data", create_if_missing=True)

data_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "unzip")
    .pip_install(
        "google-genai",
        "opencv-python-headless",
        "Pillow",
        "kaggle",
    )
)

# GPU image for YOLO26x-pose (needs CUDA)
yolo_gpu_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04", add_python="3.11"
    )
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        "ultralytics",
        "opencv-python-headless",
        "torch",
        "torchvision",
    )
)

# Sports to EXCLUDE — no useful single-person pose data
EXCLUDED_SPORTS = {
    "chess",        # board game, no movement
    "formula1",     # cars, not people
    "motogp",       # motorcycles, riders in full gear
    "hockey",       # too much gear, cluttered ice
    "ice hockey",   # same as hockey
}

# All Kaggle datasets to download for maximum coverage
KAGGLE_DATASETS = [
    {"slug": "rishikeshkonapure/sports-image-dataset", "target": "/data/sports"},
    {"slug": "niharika41298/yoga-poses-dataset", "target": "/data/yoga"},
    {"slug": "hasyimabdillah/workoutexercises-images", "target": "/data/workout"},
    {"slug": "gpiosenern/musical-instruments-classification-dataset",
        "target": "/data/instruments"},
]

# Prompt for Gemini distillation (structured coaching JSON)
COACHING_PROMPT_TEMPLATE = """You are MotionCoach, a supportive and encouraging AI movement coach.
Activity: {activity}
User's goal: improve my {activity} form

{pose_text}

All coordinates are normalized floats [0.0, 1.0] where (0,0)=top-left.

=== OUTPUT RULES ===
1. Return 1-3 feedback_points evaluating the person's form.
2. positive_note: genuinely highlight something good.
3. progress_score: 0-100 rating of form quality.
4. For ANGLE_CORRECTION visuals include exactly 2 vectors starting from
   the SAME joint: one "Current" (red), one "Target" (green).
5. Vector coordinates MUST be close to the YOLO keypoints above.
6. Always specify LEFT or RIGHT for body parts.
7. Output ONLY valid JSON matching this schema:
{{
  "status": "success",
  "error_message": null,
  "positive_note": "...",
  "progress_score": 60,
  "improvement_delta": null,
  "feedback_points": [
    {{
      "mistake_timestamp_ms": 0,
      "coaching_script": "...",
      "visuals": {{
        "overlay_type": "ANGLE_CORRECTION",
        "focus_point": {{"x": 0.5, "y": 0.5}},
        "vectors": [
          {{"start": [0.5, 0.5], "end": [0.5, 0.6],
              "color": "red", "label": "Current"}},
          {{"start": [0.5, 0.5], "end": [0.6, 0.5],
              "color": "green", "label": "Target"}}
        ]
      }}
    }}
  ]
}}
"""

# ---------------------------------------------------------------------------
# Step 1 — Download all Kaggle datasets to the Modal volume
# ---------------------------------------------------------------------------


@app.function(
    image=data_image,
    volumes={"/data": vol},
    timeout=1200,
    secrets=[modal.Secret.from_name("kaggle-secret")],
)
def download_kaggle_dataset():
    """Download + extract multiple Kaggle datasets for pose training data."""
    import kaggle
    kaggle.api.authenticate()

    for ds in KAGGLE_DATASETS:
        target = ds["target"]
        slug = ds["slug"]

        if os.path.exists(target) and len(os.listdir(target)) > 0:
            print(f"✅ {slug} already at {target}, skipping.")
            continue

        print(f"⬇️  Downloading {slug} → {target} …")
        os.makedirs(target, exist_ok=True)

        try:
            kaggle.api.dataset_download_files(slug, path=target, unzip=True)
            # Count downloaded files
            total = sum(len(f) for _, _, f in os.walk(target))
            print(f"   ✅ {slug}: {total} files downloaded")
        except Exception as e:
            print(f"   ⚠️  Failed to download {slug}: {e}")

    vol.commit()
    print("✅ All datasets downloaded and committed to volume.")


# ---------------------------------------------------------------------------
# Step 2a — Discover image paths per sport (CPU, fast)
# ---------------------------------------------------------------------------
@app.function(image=data_image, volumes={"/data": vol}, timeout=120)
def discover_images(per_sport: int = 60) -> list:
    """
    Walk the dataset and return a list of per-sport batches for YOLO filtering.
    Excludes non-pose sports. Shuffles images for variety.
    """
    import random

    # Search across ALL downloaded datasets
    data_roots = ["/data/sports", "/data/yoga",
                  "/data/workout", "/data/instruments"]

    sport_dirs = {}
    for base in data_roots:
        if not os.path.exists(base):
            print(f"   ⏭️  {base} not found, skipping")
            continue
        for root, _, files in os.walk(base):
            images = [
                os.path.join(root, f) for f in files
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if images:
                sport_name = os.path.basename(root).replace("_", " ")
                if sport_name.lower() in EXCLUDED_SPORTS:
                    print(
                        f"   ⏭️  Skipping '{sport_name}' (not useful for pose)")
                    continue
                if sport_name not in sport_dirs:
                    sport_dirs[sport_name] = []
                sport_dirs[sport_name].extend(images)

    if not sport_dirs:
        print("❌ No sport categories found.")
        return []

    # Shuffle within each sport for variety
    for imgs in sport_dirs.values():
        random.shuffle(imgs)

    print(f"📂 Found {len(sport_dirs)} pose-relevant sports:")
    batches = []
    for sport, imgs in sorted(sport_dirs.items()):
        # Oversample 4x since YOLO will reject many non-single-person images
        sample_pool = imgs[:per_sport * 4]
        print(f"   {sport}: {len(imgs)} total, scanning up to {len(sample_pool)}")
        batches.append({
            "sport": sport,
            "paths": sample_pool,
            "target": per_sport,
        })

    return batches


# ---------------------------------------------------------------------------
# Step 2b — YOLO filter one sport's images (GPU, runs in parallel)
# ---------------------------------------------------------------------------
@app.function(
    image=yolo_gpu_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=600,
    max_containers=8,
)
def yolo_filter_sport(batch: dict) -> list:
    """
    Run YOLO26x-pose on images for ONE sport category.
    Returns accepted images with pose keypoints.
    Each sport gets its own T4 container → all sports run in parallel.
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO("yolo26x-pose.pt")

    KEYPOINT_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    sport = batch["sport"]
    paths = batch["paths"]
    target = batch["target"]

    accepted = []
    skipped_no_person = 0
    skipped_multi = 0
    scanned = 0

    print(
        f"🏃 [{sport}] Starting YOLO scan on {len(paths)} images (target: {target}) …")

    for fpath in paths:
        if len(accepted) >= target:
            break

        scanned += 1

        try:
            # Read and convert to RGB to handle grayscale PNGs
            img = cv2.imread(fpath)
            if img is None:
                continue
            if len(img.shape) == 2:  # grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            results = model(img, conf=0.25, verbose=False)
        except Exception:
            continue

        r = results[0]

        # --- FILTER: exactly 1 person ---
        if r.keypoints is None or r.keypoints.xyn is None:
            skipped_no_person += 1
            continue
        num_people = r.keypoints.xyn.shape[0]
        if num_people == 0:
            skipped_no_person += 1
            continue
        if num_people > 1:
            skipped_multi += 1
            continue

        # --- FILTER: at least 3 visible keypoints (conf > 0.3) ---
        person_kps = r.keypoints.xyn[0]
        confs = r.keypoints.conf[0] if r.keypoints.conf is not None else None
        visible = 0
        pose_lines = []
        for k_idx, kp in enumerate(person_kps):
            x, y = float(kp[0]), float(kp[1])
            c = float(confs[k_idx]) if confs is not None else 1.0
            if c > 0.3 and (x > 0 or y > 0):
                visible += 1
                pose_lines.append(
                    f"  {KEYPOINT_NAMES[k_idx]}: [{x:.3f}, {y:.3f}]")

        if visible < 3:
            skipped_no_person += 1
            continue

        pose_text = "=== DETECTED POSE KEYPOINTS ===\n" + "\n".join(pose_lines)
        accepted.append({
            "path": fpath,
            "activity": sport,
            "pose_text": pose_text,
        })

    print(f"✅ [{sport}] Done: {scanned} scanned → {len(accepted)} accepted, "
          f"{skipped_no_person} no/few person, {skipped_multi} multi-person")
    return accepted


def _valid_coaching_json(parsed: dict) -> bool:
    """Check that parsed dict has required MotionCoach schema fields."""
    return (
        isinstance(parsed, dict)
        and "feedback_points" in parsed
        and "progress_score" in parsed
        and isinstance(parsed.get("feedback_points"), list)
    )


# ---------------------------------------------------------------------------
# Step 3 — Generate coaching labels with Gemini 3.0 Flash (thinking)
# ---------------------------------------------------------------------------
@app.function(
    image=data_image,
    volumes={"/data": vol},
    timeout=600,
    max_containers=50,
    secrets=[modal.Secret.from_name("gemini-api-key")],
    retries=1,
)
def label_single_image(image_info: dict) -> dict:
    """Call Gemini to produce one training row (distillation)."""
    from google import genai
    import time
    import PIL.Image

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options={"api_version": "v1beta"},
    )

    image_path = image_info["path"]
    activity = image_info["activity"]
    pose_text = image_info["pose_text"]
    prompt = COACHING_PROMPT_TEMPLATE.format(
        activity=activity, pose_text=pose_text
    )

    try:
        img = PIL.Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((768, 768))

        max_retries = 4
        text = None
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[prompt, img],
                    config={
                        "response_mime_type": "application/json",
                        "thinking_config": {"thinking_level": "low"},
                    },
                )
                text = response.text.strip()
                break
            except Exception as api_err:
                err_str = str(api_err)
                if attempt < max_retries and (
                    "429" in err_str or "503" in err_str
                    or "UNAVAILABLE" in err_str or "RESOURCE_EXHAUSTED" in err_str
                ):
                    wait = 10 * (2 ** attempt)
                    print(
                        f"   ⏳ [{os.path.basename(image_path)}] Rate limited, retry {attempt+1}/{max_retries} in {wait}s …"
                    )
                    time.sleep(wait)
                else:
                    raise

        if not text:
            return {
                "success": False,
                "file": os.path.basename(image_path),
                "error": "All retries exhausted",
            }

        parsed = json.loads(text)
        if not _valid_coaching_json(parsed):
            return {
                "success": False,
                "file": os.path.basename(image_path),
                "error": "Invalid coaching JSON from Gemini",
            }

        system_msg = "You are MotionCoach, a supportive and encouraging AI movement coach."
        user_msg = (
            f"<image>{image_path}</image>\n"
            f"Analyze the user's body mechanics for {activity}.\n\n{pose_text}"
        )
        return {
            "success": True,
            "file": os.path.basename(image_path),
            "row": {
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": text},
                ],
            },
        }
    except Exception as e:
        return {"success": False, "file": os.path.basename(image_path), "error": str(e)}


# ---------------------------------------------------------------------------
# Step 4 — Save final dataset to volume
# ---------------------------------------------------------------------------
@app.function(image=data_image, volumes={"/data": vol}, timeout=60)
def save_dataset_to_volume(rows_json: str):
    """Write the finished JSONL dataset to the volume."""
    out_path = "/data/pose_dataset.jsonl"
    with open(out_path, "w") as f:
        f.write(rows_json)
    vol.commit()
    print(f"✅ Saved dataset to volume at {out_path}")


# ---------------------------------------------------------------------------
# Entrypoint — orchestrate the full pipeline
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(max_samples: int = 2500):
    """
    Run the full data pipeline:
      1. Download Kaggle dataset (if not cached)
      2. Discover sport categories + YOLO-filter IN PARALLEL (one T4 per sport)
      3. Label in parallel with Gemini 3.0 Flash (thinking)
      4. Save JSONL to volume + local file
    """
    print("=" * 60)
    print(" MotionCoach Dataset Generator (PARALLELIZED)")
    print("=" * 60)

    # 1) Download
    print("\n📥 Step 1/4 — Ensuring Kaggle dataset is on Modal volume …")
    download_kaggle_dataset.remote()

    import json
    import os

    # 2) Discover sport categories & Check YOLO Cache
    yolo_cache_path = "yolo_cache.json"
    if os.path.exists(yolo_cache_path):
        print("\n🔍 Step 2/4 — YOLO cache found! Loading pre-filtered images …")
        with open(yolo_cache_path, "r") as f:
            all_accepted = json.load(f)
    else:
        print(
            f"\n🔍 Step 2/4 — Discovering sports & YOLO-filtering (target: {max_samples}) …")

        # Calculate per-sport target based on number of sports
        # We'll discover first, then distribute. Assume ~17 sports after exclusions.
        estimated_sports = 17
        per_sport = max(1, max_samples // estimated_sports)

        batches = discover_images.remote(per_sport)
        if not batches:
            print("❌ No sport categories found. Check the dataset.")
            return

        actual_sports = len(batches)
        per_sport = max(1, max_samples // actual_sports)
        # Update targets in batches
        for b in batches:
            b["target"] = per_sport

        print(
            f"\n🚀 Launching {actual_sports} parallel YOLO containers (T4 each) …")
        print(
            f"   Target: ~{per_sport} images per sport = ~{per_sport * actual_sports} total")

        # Fan out: one T4 container per sport, all running simultaneously
        all_accepted = []
        for sport_results in yolo_filter_sport.map(batches):
            all_accepted.extend(sport_results)
            print(
                f"   📦 Received {len(sport_results)} images, total: {len(all_accepted)}")

        if not all_accepted:
            print("❌ No usable single-person images found. Check the dataset.")
            return

        # Trim to exact target if we got more
        if len(all_accepted) > max_samples:
            all_accepted = all_accepted[:max_samples]

        print(f"\n   → {len(all_accepted)} images passed the filter.")

        # Print per-sport breakdown
        sport_counts = {}
        for img in all_accepted:
            sport_counts[img["activity"]] = sport_counts.get(
                img["activity"], 0) + 1
        print("📊 Per-sport breakdown:")
        for sport, count in sorted(sport_counts.items(), key=lambda x: -x[1]):
            print(f"   {sport}: {count}")

        # Save cache
        if not os.path.exists(yolo_cache_path):
            with open(yolo_cache_path, "w") as f:
                json.dump(all_accepted, f)
            print(f"   💾 Saved YOLO results to {yolo_cache_path}")

    # 3) Label in parallel with Gemini
    print(f"\n🤖 Step 3/4 — Labelling {len(all_accepted)} images with Gemini …")
    dataset_rows = []
    success = 0
    fail = 0

    for result in label_single_image.map(all_accepted):
        if result["success"]:
            dataset_rows.append(result["row"])
            success += 1
            if success % 25 == 0:
                print(f"   ✅ {success} labelled …")
        else:
            fail += 1
            if fail <= 5:
                print(f"   ⚠️  {result['file']}: {result['error']}")

    print(f"\n   Done: {success} success, {fail} failed.")

    if not dataset_rows:
        print("❌ No rows generated. Exiting.")
        return

    # 4) Save
    print("\n💾 Step 4/4 — Saving dataset …")
    jsonl_str = "\n".join(json.dumps(row) for row in dataset_rows)

    save_dataset_to_volume.remote(jsonl_str)

    local_path = "pose_dataset.jsonl"
    with open(local_path, "w") as f:
        f.write(jsonl_str)

    print(f"\n🎉 Generated {len(dataset_rows)} training examples.")
    print(f"   Local:  {os.path.abspath(local_path)}")
    print(f"   Volume: /data/pose_dataset.jsonl")
    print(f"\n   Next step: uv run modal run training.py")
