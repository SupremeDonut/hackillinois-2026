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
#   4. Stitch together the `video_analyzer` and `text_to_speech` functions.
"""

import modal
from fastapi import FastAPI, UploadFile, Form
from typing import Annotated

image = modal.Image.debian_slim().uv_pip_install("fastapi[standard]")
app = modal.App("motion-coach")
web_app = FastAPI()


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
    video_analyzer = modal.Cls.from_name("biomechanics-ai", "VideoAnalyzer")()
    analysis = await video_analyzer.analyze.remote.aio(
        video_bytes, user_description, activity_type
    )

    # Generate Audio Base64
    tts_worker = modal.Cls.from_name("biomechanics-ai", "TextToSpeech")()
    audio = await tts_worker.speak.remote.aio(analysis.get("coaching_script", ""))

    # Combine and return to React Native
    return {
        "status": "success",
        "analysis": analysis,
        "visuals": {},
        "audio": audio,
    }


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


# To test locally during the hackathon, run: `modal serve backend/main.py`
