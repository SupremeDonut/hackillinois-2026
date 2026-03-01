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
