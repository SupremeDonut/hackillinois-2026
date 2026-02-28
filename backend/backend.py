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
        # skip old-format weights, use safetensors
        ignore_patterns=["*.pt", "*.bin"],
    )


video_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git")
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
    # Use two B200s for the massive 122B model
    gpu="B200",
    image=video_image,
    timeout=600,  # Max 10 mins per analysis
    ephemeral_disk=512 * 1024,  # Minimum 512 GiB required by Modal
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class VideoAnalyzer:
    @modal.enter()
    def load_model(self):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        # Weights are baked into the image at MODEL_CACHE_DIR — no download at runtime
        self.raw_model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_CACHE_DIR,
            torch_dtype="auto",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_CACHE_DIR)

    @modal.method()
    def analyze(self, video_bytes: bytes, user_description: str, activity_type: str, previous_analysis: str = ""):
        import json

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            prompt = f"Act as a professional coach for {activity_type}. The user's goal is: {user_description}. "
            if previous_analysis:
                prompt += f"\n\nHere is the context from the user's PREVIOUS attempt:\n{previous_analysis}\n\nUse this to compute the improvement_delta and avoid repeating the exact same feedback if they improved."

            prompt += "\n\nProvide coaching feedback. All coordinates for visuals MUST be relative floats between 0.0 and 1.0 (e.g. [0.4, 0.5]), where [0.0, 0.0] is top-left and [1.0, 1.0] is bottom-right."

            schema_json = BiomechanicalAnalysisLLM.model_json_schema()
            prompt += f"\n\nOutput ONLY a valid JSON object matching this JSON Schema:\n{json.dumps(schema_json)}"

            # --- Frame extraction with decord ---
            import numpy as np
            from decord import VideoReader, cpu as decord_cpu
            from PIL import Image as PILImage

            vr = VideoReader(tmp.name, ctx=decord_cpu(0))
            actual_fps = vr.get_avg_fps()
            total_frames = len(vr)
            total_duration = total_frames / actual_fps

            # B200 has 180GB VRAM — sample 24 fps for first 5 seconds
            end_time = min(5.0, total_duration)
            nframes = max(1, int(end_time * 24.0))
            frame_indices = np.linspace(0, min(int(end_time * actual_fps), total_frames) - 1,
                                        num=nframes, dtype=int).tolist()
            raw_frames = vr.get_batch(frame_indices).asnumpy()
            video_frames = [PILImage.fromarray(f) for f in raw_frames]
            del vr

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

            # === Official Qwen3-VL pipeline (from model card) ===
            # Uses processor.apply_chat_template with tokenize=True in ONE step.
            # This is DIFFERENT from Qwen2.5-VL which used a two-step process:
            #   1. apply_chat_template(tokenize=False) + process_vision_info()
            #   2. processor(text=..., videos=...)
            # The one-step approach correctly generates mm_token_type_ids and
            # splits video_grid_thw per-frame as Qwen3-VL requires.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda")

            # Debug: log processor output
            print(f"[DEBUG] Processor output keys: {list(inputs.keys())}")
            for k, v in inputs.items():
                if hasattr(v, 'shape'):
                    print(f"[DEBUG]   {k}: shape={v.shape}, dtype={v.dtype}")

            # Deep debug: count type-2 groups in mm_token_type_ids
            import itertools
            import torch
            mm_types = inputs["mm_token_type_ids"][0].tolist()
            type_groups = [(k, len(list(g)))
                           for k, g in itertools.groupby(mm_types)]
            video_groups = [(k, l) for k, l in type_groups if k == 2]
            print(f"[DEBUG] Total modality groups: {len(type_groups)}")
            print(f"[DEBUG] Video (type=2) groups: {len(video_groups)}")
            print(f"[DEBUG] video_grid_thw values: {inputs['video_grid_thw']}")
            print(
                f"[DEBUG] video_grid_thw entries: {inputs['video_grid_thw'].shape[0]}")

            # FIX: Qwen3-VL creates per-frame <|vision_start|>...<|vision_end|> blocks,
            # so each frame is a separate type-2 group needing its own grid_thw entry.
            # The processor outputs one [T, H, W] for the whole video — we need to
            # expand it to num_frame_groups entries of [1, H, W] each.
            n_video_groups = len(video_groups)
            n_grid_entries = inputs["video_grid_thw"].shape[0]
            if n_video_groups > n_grid_entries:
                print(
                    f"[DEBUG] MISMATCH: {n_video_groups} video groups but {n_grid_entries} grid entries. Expanding...")
                orig_thw = inputs["video_grid_thw"][0]  # [T, H, W]
                t_val, h_val, w_val = orig_thw[0].item(
                ), orig_thw[1].item(), orig_thw[2].item()
                # Each frame group gets [1, H, W]
                expanded = torch.tensor(
                    [[1, h_val, w_val]] * n_video_groups, dtype=orig_thw.dtype, device=orig_thw.device)
                inputs["video_grid_thw"] = expanded
                print(
                    f"[DEBUG] Expanded video_grid_thw to shape: {expanded.shape}")

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

    # 2. Call the VL model
    try:
        # We need to await the remote call if we make the router async
        llm_response = await VideoAnalyzer().analyze.remote.aio(
            video_bytes=video_bytes,
            user_description=user_description,
            activity_type=activity_type,
            previous_analysis=previous_analysis_str
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

    # 3. Generate TTS for all feedback points concurrently
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

    # 4. Construct Final Response
    final_feedback_points = []
    for fp, audio_b64 in zip(feedback_points, audio_results):
        fp_copy = fp.copy()
        fp_copy["audio_url"] = f"data:audio/wav;base64,{audio_b64}"
        final_feedback_points.append(fp_copy)

    llm_response["feedback_points"] = final_feedback_points

    return Response(content=json.dumps(llm_response), media_type="application/json")
