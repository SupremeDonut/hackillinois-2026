"""
training.py — LoRA fine-tuning Qwen3-VL-32B on Modal (8x H200)
Usage: cd backend/modal_fine_tuning && uv run modal run training.py
"""

import modal

app = modal.App("qwen3-vl-finetune")
vol = modal.Volume.from_name("qwen-finetune-data", create_if_missing=True)

training_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch", "torchvision", "transformers", "accelerate", "peft",
        "qwen-vl-utils", "wheel", "Pillow",
        "huggingface_hub", "decord",
    )
    .run_commands("pip install 'ms-swift[llm]' -U")
)

NUM_GPUS = 8


@app.function(image=training_image, volumes={"/data": vol}, timeout=60)
def verify_dataset():
    import os
    path = "/data/pose_dataset.jsonl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run data_acquisition.py first.")
    num_lines = sum(1 for _ in open(path))
    print(f"Dataset: {num_lines} examples, {os.path.getsize(path) / 1024:.0f} KB")
    return num_lines


@app.function(
    image=training_image,
    gpu=f"H200:{NUM_GPUS}",
    timeout=14400,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train(num_examples: int = 0):
    import subprocess, sys, os, json, re

    os.environ["USE_HF"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "1"

    raw_path = "/data/pose_dataset.jsonl"
    output_dir = "/data/checkpoints"
    train_path = "/tmp/pose_train.jsonl"
    val_path = "/tmp/pose_val.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    import random
    all_rows = []
    with open(raw_path) as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "images" not in row and "messages" in row:
                for m in row["messages"]:
                    if m.get("role") == "user" and isinstance(m.get("content"), str):
                        match = re.search(r"<image>(.+?)</image>", m["content"])
                        if match:
                            row["images"] = [match.group(1).strip()]
                            break
            all_rows.append(row)

    max_tok = 4096
    max_chars = max_tok * 3  # conservative ~3 chars/token for mixed content
    before = len(all_rows)
    all_rows = [
        row for row in all_rows
        if sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str)
            else sum(len(p.get("text", "")) for p in m["content"] if isinstance(p, dict))
            for m in row.get("messages", [])
        ) <= max_chars
    ]
    print(f"Filtered: {before} → {len(all_rows)} rows (dropped {before - len(all_rows)} rows exceeding ~{max_tok} tokens)")

    random.seed(42)
    random.shuffle(all_rows)
    val_size = max(50, len(all_rows) // 10)
    val_rows = all_rows[:val_size]
    train_rows = all_rows[val_size:]

    with open(train_path, "w") as f:
        for row in train_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for row in val_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Split: {len(train_rows)} train, {len(val_rows)} val")

    command = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "--master_port=29500",
        "/usr/local/bin/swift", "sft",
        "--model", "Qwen/Qwen3-VL-32B-Instruct",
        "--dataset", train_path,
        "--val_dataset", val_path,
        "--train_type", "lora",
        "--lora_rank", "256",
        "--lora_alpha", "512",
        "--lora_dropout", "0.05",
        "--num_train_epochs", "10",
        "--learning_rate", "1e-4",
        "--lr_scheduler_type", "cosine",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "4",
        "--warmup_ratio", "0.1",
        "--weight_decay", "0.01",
        "--bf16", "true",
        "--attn_impl", "sdpa",
        "--padding_free", "false",
        "--packing", "false",
        "--freeze_vit", "false",
        "--freeze_aligner", "false",
        "--gradient_checkpointing", "true",
        "--max_length", "4096",
        "--output_dir", output_dir,
        "--save_steps", "25",
        "--save_total_limit", "5",
        "--eval_steps", "25",
        "--logging_steps", "5",
    ]

    print(f"$ {' '.join(command)}\n")

    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(line)
    sys.stdout.flush()
    process.wait()

    if process.returncode != 0:
        sys.exit(process.returncode)

    vol.commit()
    print("\nTraining complete!")

    run_dirs = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not run_dirs:
        return None
    latest_run = max(run_dirs, key=os.path.getmtime)
    checkpoints = [
        d for d in os.listdir(latest_run)
        if os.path.isdir(os.path.join(latest_run, d)) and d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    latest_ckpt = max(checkpoints, key=lambda x: int(x.replace("checkpoint-", "") or "0"))
    latest_ckpt_path = os.path.join(latest_run, latest_ckpt)

    repo_id = os.environ.get("HF_ADAPTER_REPO_ID", "").strip()
    token = os.environ.get("HF_TOKEN")
    if repo_id and token:
        try:
            readme_path = os.path.join(latest_ckpt_path, "README.md")
            if os.path.exists(readme_path):
                with open(readme_path) as f:
                    readme = f.read()
                readme = re.sub(
                    r"base_model:\s*.+",
                    "base_model: Qwen/Qwen3-VL-32B-Instruct",
                    readme,
                )
                with open(readme_path, "w") as f:
                    f.write(readme)

            from huggingface_hub import HfApi
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            api.upload_folder(
                folder_path=latest_ckpt_path, repo_id=repo_id,
                commit_message=f"Upload LoRA adapter from {latest_ckpt}",
            )
            print(f"Adapter uploaded: https://huggingface.co/{repo_id}")
        except Exception as e:
            print(f"HF upload failed: {e}")

    return latest_ckpt_path


@app.function(image=training_image, volumes={"/data": vol}, timeout=300)
def package_checkpoint(ckpt_path: str) -> bytes:
    """Tar-gz the checkpoint and return bytes if under 2 GB."""
    import tarfile, io, os

    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(ckpt_path) for f in fns
    )
    size_mb = total_bytes / (1024 * 1024)
    print(f"Checkpoint size: {size_mb:.0f} MB ({ckpt_path})")

    if total_bytes > 2 * 1024**3:
        print(f"Too large to return ({size_mb:.0f} MB > 2048 MB). Use: modal volume get qwen-finetune-data {ckpt_path}/ ./local-checkpoint/")
        return b""

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(ckpt_path, arcname=os.path.basename(ckpt_path))
    data = buf.getvalue()
    print(f"Compressed: {len(data) / (1024*1024):.0f} MB")
    return data


upload_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")


@app.function(
    image=upload_image,
    volumes={"/data": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=1800,
)
def _upload_to_hf(ckpt_path: str, repo_id: str):
    import os, shutil, tempfile
    from huggingface_hub import HfApi

    KEEP = {"adapter_model.safetensors", "adapter_config.json", "additional_config.json"}

    staging = tempfile.mkdtemp()
    for fn in os.listdir(ckpt_path):
        if fn in KEEP:
            src = os.path.join(ckpt_path, fn)
            shutil.copy2(src, os.path.join(staging, fn))

    readme = f"""\
---
base_model: Qwen/Qwen3-VL-32B-Instruct
tags:
  - peft
  - lora
  - qwen3-vl
library_name: peft
---

# MotionCoach — Qwen3-VL-32B LoRA Adapter

Fine-tuned LoRA adapter for pose/form correction on exercise images.

- **Base model:** [Qwen/Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
- **Checkpoint:** {os.path.basename(ckpt_path)}
"""
    with open(os.path.join(staging, "README.md"), "w") as f:
        f.write(readme)

    files = []
    for fn in sorted(os.listdir(staging)):
        size = os.path.getsize(os.path.join(staging, fn)) / (1024 * 1024)
        files.append((fn, size))
    print(f"Uploading {len(files)} files from {ckpt_path}:")
    for fn, size in files:
        print(f"  {fn}  ({size:.1f} MB)")

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=staging,
        repo_id=repo_id,
        commit_message=f"Upload LoRA adapter from {os.path.basename(ckpt_path)}",
    )
    shutil.rmtree(staging)
    print(f"\nUploaded: https://huggingface.co/{repo_id}")


@app.local_entrypoint(name="upload_checkpoint")
def upload_checkpoint():
    import os

    ckpt_path = os.environ.get("CKPT_PATH", "/data/checkpoints/v18-20260301-051939/checkpoint-100")
    repo_id = os.environ.get("HF_ADAPTER_REPO_ID", "Playbird12/motioncoach-qwen3vl-32b-lora").strip()

    print(f"Uploading {ckpt_path} → https://huggingface.co/{repo_id}")
    _upload_to_hf.remote(ckpt_path, repo_id)


@app.local_entrypoint()
def main():
    import tarfile, io, os

    num_examples = verify_dataset.remote()
    print(f"Launching training on {NUM_GPUS}x H200 ({num_examples} examples)...")
    ckpt_path = train.remote(num_examples)

    if not ckpt_path:
        print("No checkpoint produced.")
        return

    print(f"\nPackaging checkpoint: {ckpt_path}")
    data = package_checkpoint.remote(ckpt_path)

    if not data:
        print("Checkpoint too large to download. Use:")
        print(f"  modal volume get qwen-finetune-data {ckpt_path}/ ./lora-adapter/")
        return

    out_dir = "lora-adapter"
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(out_dir)

    print(f"\nLoRA adapter saved to: {os.path.abspath(out_dir)}/")
    for f in sorted(os.listdir(out_dir)):
        sub = os.path.join(out_dir, f)
        if os.path.isdir(sub):
            for sf in sorted(os.listdir(sub)):
                size = os.path.getsize(os.path.join(sub, sf))
                print(f"  {f}/{sf}  ({size / (1024*1024):.1f} MB)")
        else:
            size = os.path.getsize(sub)
            print(f"  {f}  ({size / (1024*1024):.1f} MB)")
