# MotionCoach Fine-Tuning Pipeline

Fine-tune **Qwen3-VL-32B-Instruct** for pose correction coaching using Modal H200 GPUs.

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

1. Download multiple Kaggle datasets (sports, yoga, workout) to a Modal volume
2. Run **YOLO26x-pose** on every image and **filter out** images with 0 people or >1 person
3. Send each single-person image to **Gemini 3.0 Flash** to generate structured coaching JSON labels (distillation)
4. Save `pose_dataset.jsonl` locally **and** to the Modal volume

To limit the number of samples (default 2500):

```bash
uv run modal run data_acquisition.py --max-samples 500
```

### Step 2 — Fine-tune the model (~30-90 min)

```bash
uv run modal run training.py
```

This will:

1. Verify the dataset exists on the volume
2. Split 90/10 into train/val sets for overfitting detection
3. Spin up **8× H200 GPUs** (141 GB VRAM each) and run **MS-Swift LoRA SFT** with DDP
4. Save checkpoints every 25 steps and evaluate on the validation set
5. Download the LoRA adapter to `./lora-adapter/` locally (if under 2 GB)
6. Optionally upload to Hugging Face Hub if `HF_ADAPTER_REPO_ID` is set

**Training configuration:**

| Setting | Value |
|---|---|
| LoRA rank / alpha | 256 / 512 |
| LoRA dropout | 0.05 |
| Epochs | 10 |
| Learning rate | 1e-4 (cosine schedule) |
| Effective batch size | 64 (2/device × 4 grad accum × 8 GPUs) |
| Vision encoder (VIT) | Unfrozen |
| Aligner | Unfrozen |
| Precision | bf16 |
| Attention | SDPA |

### Step 3 — Use the fine-tuned model

**Option A — Load from local adapter** (downloaded automatically after training):

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

base = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
model = PeftModel.from_pretrained(base, "./lora-adapter/checkpoint-XXX")
```

**Option B — Load from Hugging Face** (if `HF_ADAPTER_REPO_ID` was configured):

```python
model = PeftModel.from_pretrained(base, "your-username/motioncoach-qwen3vl-32b-lora")
```

**Option C — Download from Modal volume manually:**

```bash
modal volume ls qwen-finetune-data /checkpoints/
modal volume get qwen-finetune-data /checkpoints/<run-dir>/<checkpoint>/ ./lora-adapter/
```

## File Overview

| File | Purpose |
|---|---|
| `data_acquisition.py` | Downloads Kaggle data, YOLO-filters, labels with Gemini 3.0 Flash, saves JSONL |
| `training.py` | Runs LoRA fine-tuning on 8× H200 via MS-Swift, downloads adapter locally |
| `pose_dataset.jsonl` | Generated training data (~2300 pose coaching examples) |
| `yolo_cache.json` | Cached YOLO filter results to skip re-processing |
