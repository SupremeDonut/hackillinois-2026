# MotionCoach Fine-Tuning Pipeline

Fine-tune **Qwen3-VL-32B-Instruct** for pose correction coaching using Modal B200 GPUs.

## Quick Start

### Prerequisites (one-time setup)

```bash
# 1. Create Modal secrets
modal secret create gemini-api-key   GEMINI_API_KEY=<your-gemini-key>
modal secret create kaggle-secret    KAGGLE_USERNAME=<user> KAGGLE_KEY=<key>
modal secret create huggingface-secret HF_TOKEN=<your-hf-token>
# Optional: add OpenRouter for dual-model consensus (free tier ~200 req/day)
# modal secret create gemini-api-key GEMINI_API_KEY=... OPENROUTER_API_KEY=<your-openrouter-key>
# Optional: upload adapter to Hugging Face after training
# modal secret create huggingface-secret HF_TOKEN=<token> HF_ADAPTER_REPO_ID=<your-username>/motioncoach-qwen3vl-32b-lora
```

> **Kaggle key**: Go to [kaggle.com/settings](https://www.kaggle.com/settings) → "Create New API Token" → open the downloaded `kaggle.json`.
>
> **HF token**: Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → create a token with `read` access (Qwen3-VL is gated). For upload, use a token with `write` access.

### Step 1 — Generate the dataset (~15-30 min)

```bash
cd backend/modal_fine_tuning
uv run modal run data_acquisition.py
```

This will:

1. Download the [Sports Image Dataset](https://www.kaggle.com/datasets/rishikeshkonapure/sports-image-dataset) (22 sports) to a Modal volume
2. Run **YOLO pose** on every image and **filter out** images with 0 people or >1 person
3. Send each single-person image to **Gemini** (and optionally **OpenRouter**) to generate coaching JSON labels; when both are used, consensus (same score band) yields a higher-confidence training set
4. Save `pose_dataset.jsonl` locally **and** to the Modal volume

To limit the number of samples (default 500):

```bash
uv run modal run data_acquisition.py --max-samples 200
```

### Step 2 — Fine-tune the model (~1-3 hours)

```bash
uv run modal run training.py
```

This will:

1. Verify the dataset exists on the volume
2. Spin up **8× B200 GPUs** and run **MS-Swift LoRA SFT** with DeepSpeed ZeRO-2
3. Save checkpoints to `/data/checkpoints` on the volume
4. If `HF_ADAPTER_REPO_ID` is set in the Hugging Face secret, upload the **latest checkpoint** to that Hub repo (e.g. `your-username/motioncoach-qwen3vl-32b-lora`)

### Step 3 — Use the fine-tuned model

**Option A — Load adapter from Hugging Face** (if you set `HF_ADAPTER_REPO_ID` and the upload succeeded):

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-32B-Instruct", token=...)
model = PeftModel.from_pretrained(base, "your-username/motioncoach-qwen3vl-32b-lora", token=...)
```

**Option B — Load adapter from local/volume path** (e.g. after downloading or mounting the checkpoints):

```python
model = PeftModel.from_pretrained(base, "/path/to/checkpoint-150")
```

## File Overview

| File | Purpose |
|---|---|
| `data_acquisition.py` | Downloads Kaggle data, YOLO-filters, labels with Gemini (+ optional OpenRouter), saves JSONL |
| `training.py` | Runs LoRA fine-tuning on Modal B200 via MS-Swift |
