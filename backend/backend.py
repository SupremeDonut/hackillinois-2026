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

video_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
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
    visuals: VisualOverlay


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
        import outlines

        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = outlines.from_transformers(
            self.raw_model, self.processor.tokenizer
        )

    @modal.method()
    def analyze(self, video_bytes: bytes, user_description: str, activity_type: str, previous_analysis: str = ""):
        from qwen_vl_utils import process_vision_info

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()

            prompt = f"Act as a professional coach for {activity_type}. The user's goal is: {user_description}. "
            if previous_analysis:
                prompt += f"\n\nHere is the context from the user's PREVIOUS attempt:\n{previous_analysis}\n\nUse this to compute the improvement_delta and avoid repeating the exact same feedback if they improved."

            prompt += "\n\nProvide coaching feedback. All coordinates for visuals MUST be relative floats between 0.0 and 1.0 (e.g. [0.4, 0.5]), where [0.0, 0.0] is top-left and [1.0, 1.0] is bottom-right."

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
                return_tensors="pt",
            ).to("cuda")

            # Use Outlines to force the JSON schema
            result = self.model(
                **inputs,
                output_type=BiomechanicalAnalysisLLM,
            )
        return result.model_dump()


@app.function(image=modal.Image.debian_slim().pip_install("fastapi", "python-multipart"))
@modal.web_endpoint(method="POST")
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
    tts_engine = TextToSpeech()
    feedback_points = llm_response.get("feedback_points", [])

    # We will gather all the remote calls and wait for them
    # Note: Modal .remote() is synchronous in standard python, so we map over them using .map

    # Prepare the scripts
    scripts = [fp["coaching_script"] for fp in feedback_points]

    # Run TTS in parallel
    if scripts:
        audio_results = []
        async for res in tts_engine.speak.map.aio(scripts):
            audio_results.append(res)
    else:
        audio_results = []

    # 4. Construct Final Response
    final_feedback_points = []
    for fp, audio_b64 in zip(feedback_points, audio_results):
        fp_copy = fp.copy()
        fp_copy["audio_url"] = f"data:audio/wav;base64,{audio_b64}"
        final_feedback_points.append(fp_copy)

    llm_response["feedback_points"] = final_feedback_points

    return Response(content=json.dumps(llm_response), media_type="application/json")
