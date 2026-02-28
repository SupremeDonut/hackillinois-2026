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
        samples, sample_rate = self.model.create(phonemes, "af_heart", is_phonemes=True)

        wav_io = io.BytesIO()
        sf.write(wav_io, samples, sample_rate, format="WAV")
        return base64.b64encode(wav_io.getvalue()).decode("utf-8")


# TODO: fix schema
class TechnicalStats(BaseModel):
    observed_angle: float
    target_angle: float


class BiomechanicalAnalysis(BaseModel):
    mistake_timestamp_ms: int
    coaching_script: str
    positive_note: str
    progress_score: int
    improvement_delta: int
    technical_stats: TechnicalStats


@app.cls(
    gpu="A10G",  # Choose GPU based on model size (A10G is usually enough for 7B)
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
    def analyze(self, video_bytes: bytes, user_description: str, activity_type: str):
        from qwen_vl_utils import process_vision_info

        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
            tmp.write(video_bytes)
            tmp.flush()
            prompt = f"Coach for {activity_type}. User goal: {user_description}."
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
            image_inputs, video_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
            ).to("cuda")

            # Use Outlines to force the JSON schema
            result = self.model(
                **inputs,
                output_type=BiomechanicalAnalysis,
            )
        return result.model_dump()
