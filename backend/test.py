import base64
import httpx
import time
import json

# 1. Configuration
API_URL = "https://dywang2--motion-coach-fastapi-app-dev.modal.run/analyze"  # Change if using Modal/Cloud
VIDEO_PATH = "tests/test.mp4"


def test_video_analysis():
    print(f"🚀 Starting analysis for: {VIDEO_PATH}")
    start_time = time.time()

    # 2. Prepare the request
    # We send the metadata as query parameters and the video as a file
    form_data = {
        "activity_type": "badminton_swing",
        "user_description": "I'm struggling with a slice and need to keep my lead arm straight.",
    }
    try:
        with open(VIDEO_PATH, "rb") as video_file:
            files = {"video_file": (VIDEO_PATH, video_file, "video/mp4")}

            # 3. Execute Request
            # Increase timeout because GPU inference is not instant!
            with httpx.Client(timeout=120.0) as client:
                response = client.post(API_URL, data=form_data, files=files)

        # 4. Handle Results
        duration = time.time() - start_time
        if response.status_code == 200:
            print(f"✅ Success! (Took {duration:.2f}s)")
            print("\n--- Biomechanical Feedback ---")
            data = response.json()
            print(json.dumps(data, indent=2))

            # Save audio to mp3
            audio_b64 = data.get("audio")
            if audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
                with open("output.mp3", "wb") as f:
                    f.write(audio_bytes)
                print("\n🎵 Audio saved to output.mp3")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")

    except FileNotFoundError:
        print(f"⚠️ Error: Could not find file at {VIDEO_PATH}")
    except httpx.ConnectError:
        print("⚠️ Error: Could not connect to the server. Is it running?")


if __name__ == "__main__":
    test_video_analysis()
