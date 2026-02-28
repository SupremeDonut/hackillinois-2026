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
    dataset_path = "/tmp/pose_dataset.jsonl"
    os.makedirs(output_dir, exist_ok=True)

    # Normalize dataset for Swift
    with open(raw_path) as f_in, open(dataset_path, "w") as f_out:
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
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    epochs = "5" if num_examples < 100 else "3" if num_examples < 300 else "2"

    command = [
        "torchrun",
        f"--nproc_per_node={NUM_GPUS}",
        "--master_port=29500",
        "/usr/local/bin/swift", "sft",
        "--model", "Qwen/Qwen3-VL-32B-Instruct",
        "--dataset", dataset_path,
        "--train_type", "lora",
        "--lora_rank", "64",
        "--lora_alpha", "128",
        "--num_train_epochs", epochs,
        "--learning_rate", "1e-4",
        "--per_device_train_batch_size", "2",
        "--gradient_accumulation_steps", "2",
        "--warmup_ratio", "0.05",
        "--weight_decay", "0.01",
        "--bf16", "true",
        "--attn_impl", "sdpa",
        "--padding_free", "false",
        "--packing", "false",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--gradient_checkpointing", "true",
        "--max_length", "4096",
        "--output_dir", output_dir,
        "--save_steps", "50",
        "--save_total_limit", "3",
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

    # Upload latest checkpoint to HF Hub if configured
    run_dirs = [
        os.path.join(output_dir, d) for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    if not run_dirs:
        return
    latest_run = max(run_dirs, key=os.path.getmtime)
    checkpoints = [
        d for d in os.listdir(latest_run)
        if os.path.isdir(os.path.join(latest_run, d)) and d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return
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


@app.local_entrypoint()
def main():
    num_examples = verify_dataset.remote()
    print(f"Launching training on {NUM_GPUS}x H200 ({num_examples} examples)...")
    train.remote(num_examples)
