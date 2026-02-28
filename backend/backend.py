import modal
from pydantic import BaseModel
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
        "pydantic",
        "decord",
        "torchvision",
        "outlines",
    )
)

app = modal.App("biomechanics-ai")


# TODO: fix schema
class TechnicalStats(BaseModel):
    observed_angle: float
    target_angle: float


class BiomechanicalAnalysis(BaseModel):
    movement_analysis_log: str
    mistake_timestamp_ms: int
    coaching_script: str
    positive_note: str
    progress_score: int
    improvement_delta: int
    # technical_stats: TechnicalStats


@app.cls(
    gpu="L40S",
    image=video_image,
    timeout=600,  # Max 10 mins per analysis
    scaledown_window=600,
)
class VideoAnalyzer:
    @modal.enter()
    def load_model(self):
        # This only runs once when the container starts
        import outlines
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
        self.raw_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, max_pixels=768 * 28 * 28
        )

        # Wrap with outlines for constrained (structured) generation
        self.outlines_model = outlines.from_transformers(self.raw_model, self.processor)
        self.generator = outlines.Generator(self.outlines_model, BiomechanicalAnalysis)

    @modal.method()
    def analyze(self, video_bytes: bytes, user_description: str, activity_type: str):
        import outlines

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        prompt = f"You are a biomechanics coach for {activity_type}. User goal: {user_description}. Analyze the movement in the video and respond with a structured coaching analysis. Limit responses to 100 words or less."

        # Build a Chat input with the video file path and text prompt
        # outlines.Chat uses the HF multimodal dict format for transformers models
        chat_input = outlines.inputs.Chat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": outlines.Video(tmp_path)},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        )

        # outlines.Generator enforces the BiomechanicalAnalysis JSON schema
        # and returns a validated Pydantic model instance directly
        result: BiomechanicalAnalysis = self.generator(chat_input, max_new_tokens=512)
        print(result)
        return result
