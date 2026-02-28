"""
# ==============================================================================
# 🎯 [DEV 2] BACKEND ENTRYPOINT (modal/main.py)
# ==============================================================================
# Purpose:
#   This file is the "brain" of the application. It receives the 5-second video
#   from the React Native frontend, orchestrates the Gemini AI analysis, fetches
#   the voice-over audio from ElevenLabs, and returns the final JSON payload.
#
# Responsibilities:
#   1. Define the serverless GPU environment using Modal.
#   2. Expose a FastAPI `POST /analyze` endpoint.
#   3. Parse `multipart/form-data` requests (binary video + text fields).
#   4. Stitch together the `gemini_client` and `text_to_speech` functions.
#
# Important Constraints & Hackathon Tips:
#   - NEVER use `Base64` for the video upload; it inflates the payload by 33%
#     and will cause the React Native app to timeout. Stick to `form-data`.
#   - Return the data exactly matching the schema defined in `mock_response.json`
#     so Dev 3's frontend math doesn't break.
# ==============================================================================
"""

import modal
from fastapi import FastAPI, UploadFile, Form
from typing import Annotated

# --- Isolated Feature Modules ---
from gemini_client import analyze_video_with_gemini

app = modal.App("motion-coach")
web_app = FastAPI()

# Define the container environment and dependencies
image = modal.Image.debian_slim().pip_install_from_pyproject("pyproject.toml")


@web_app.post("/analyze")
async def analyze_endpoint(
    video_file: UploadFile,
    activity_type: Annotated[str, Form()],
    user_description: Annotated[str, Form()],
):
    """
    Receives the raw .mp4 binary from the phone and starts the AI pipeline.
    """
    # Load binary directly into memory (safe since videos are capped at 5s)
    video_bytes = await video_file.read()

    # 1. Analyze with Gemini (find the mistake and generate feedback)
    analysis_json = analyze_video_with_gemini(
        video_bytes, user_description, activity_type
    )

    # 2. Generate Audio URL (Pass the script to ElevenLabs)
    # Note: We proxy this so the ElevenLabs API key isn't leaked in the mobile app
    audio_url = text_to_speech.remote(analysis_json.get("coaching_script", ""))

    # 3. Combine and return to React Native
    return {
        "status": "success",
        "analysis": analysis_json,
        "visuals": {},  # Add visuals payload depending on Gemini output
        "audio_url": audio_url,
    }


# Bind the FastAPI app to Modal's serverless infrastructure
tts_image = modal.Image.debian_slim().uv_pip_install("kokoro_onnx", "soundfile")


@app.function(gpu="h100", image=tts_image)
async def text_to_speech(text: str):
    from kokoro_onnx import Kokoro
    import io
    import soundfile as sf
    from misaki import en, espeak

    fallback = espeak.EspeakFallback(british=False)
    g2p = en.G2P(trf=False, british=False, fallback=fallback)
    # Initialize the model (downloads weights on first run)
    model = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    phonemes, _ = g2p(text)
    samples, sample_rate = model.create(phonemes, "af_heart", is_phonemes=True)

    # Convert to WAV bytes (required by React Native Video component)
    wav_io = io.BytesIO()
    sf.write(wav_io, samples, sample_rate, format="WAV")
    wav_io.seek(0)

    # Return as base64 (React Native can decode this)
    import base64

    return base64.b64encode(wav_io.getvalue()).decode("utf-8")


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


# To test locally during the hackathon, run: `modal serve backend/main.py`
