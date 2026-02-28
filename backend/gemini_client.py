"""
# ==============================================================================
# 🧠 [DEV 2] GEMINI AI VISION PIPELINE (gemini_client.py)
# ==============================================================================
# Purpose:
#   This file handles all communication with Google's Gemini 3.1 Flash API.
#   It takes the raw video bytes, uploads them, and forces the LLM to return
#   a strict JSON object containing precise X/Y coordinates and timestamps.
#
# Responsibilities:
#   1. Initialize the `google-generativeai` SDK.
#   2. Define the Pydantic schema for "Structured Outputs" to prevent the LLM
#      from returning markdown blocks like ```json ... ``` which crash the app.
#   3. Write the "Golden Prompt" that instructs the AI to act like a coach.
#
# The Frame Sampling Warning:
#   Gemini 3.1 analyzes video natively, but it heavily downsamples the frames 
#   (often 1 frame per second). For fast motions (like a baseball pitch), it 
#   MIGHT skip the exact milliseconds the user made the mistake.
#   If you notice this happening during testing, pivot to using `ffmpeg` inside
#   this file to extract 10 frames-per-second, and send those images as a list 
#   instead of sending the raw .mp4.
# ==============================================================================
"""
import os
import json


def analyze_video_with_gemini(video_bytes: bytes, user_description: str, activity_type: str) -> dict:
    """
    Uploads the video frame to Gemini and prompts it for biomechanical feedback.

    Args:
        video_bytes: Raw binary of the .mp4 file.
        user_description: What the user is trying to achieve (e.g., "Fix my slice")
        activity_type: Enum classification (e.g., "golf_swing")

    Returns:
        dict: A strict dictionary matching the expected 'analysis' schema.
    """

    # Example Prompt Architecture (Requires optimization):
    prompt = f"""
    You are a supportive biomechanics coach for {activity_type}.
    The user wants to improve: {user_description}.
    
    Tasks:
    1. Identify the single most impactful biomechanical correction.
    2. Determine the EXACT MILLISECOND (mistake_timestamp_ms) the error happens.
    3. Determine the relative X,Y coordinates (0.0 to 1.0) of the body part in question.
    4. Provide an encouraging script (coaching_script) to speak to the user.
    """

    # 🚨 CRITICAL: Use SDK Structured Outputs (response_schema=MyPydanticModel) here!

    # Mock return so Dev 1 & Dev 3 can build the frontend immediately without
    # waiting for this prompt engineering to be finished:
    return {
        "mistake_timestamp_ms": 1450,
        "coaching_script": f"Try extending your elbow more for {activity_type}.",
        "positive_note": "Great power on the swing!",
        "progress_score": 85,
        "improvement_delta": 5,
        "technical_stats": {
            "observed_angle": 90,
            "target_angle": 120
        }
    }
