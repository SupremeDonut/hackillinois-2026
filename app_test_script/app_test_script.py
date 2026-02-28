import requests
import sys
import json


def test_api(video_path: str):
    url = "https://jonathantao--biomechanics-ai-analyze.modal.run"

    print(f"Loading video: {video_path}")

    try:
        with open(video_path, 'rb') as f:
            # Match the exact FormData fields the frontend sends
            files = {
                'video_file': ('test_video.mp4', f, 'video/mp4')
            }
            data = {
                'activity_type': 'Badmition Swing',
                'user_description': 'I am a beginner at badminton. I want to improve my swing.',
                'previous_analysis': ''
            }

            print(f"Sending POST request to {url}...")
            print("This may take a few minutes depending on the queue...")

            # The Modal endpoint might take up to 10 minutes (600 seconds)
            response = requests.post(url, files=files, data=data, timeout=600)

            print(f"\nStatus Code: {response.status_code}")

            try:
                json_data = response.json()
                print("\n=== SUCCESS: Parsed JSON Response ===")
                # Print nicely formatted JSON, but maybe truncate the base64 audio to avoid flooding the terminal
                for pt in json_data.get("feedback_points", []):
                    if "audio_url" in pt:
                        pt["audio_url"] = pt["audio_url"][:40] + \
                            "... [TRUNCATED BASE64]"

                print(json.dumps(json_data, indent=2))
            except Exception as e:
                print("\n=== ERROR: Failed to decode JSON ===")
                print("Raw response:")
                print(response.text)

    except FileNotFoundError:
        print(f"Error: Could not find video file at {video_path}")
        sys.exit(1)


if __name__ == "__main__":
    import os
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Try to find a video file in the same directory
    video_files = glob.glob(os.path.join(script_dir, "*.mp4")) + \
        glob.glob(os.path.join(script_dir, "*.mov"))

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    elif video_files:
        video_path = video_files[0]
        print(f"Automatically selected video: {os.path.basename(video_path)}")
    else:
        print("Usage: python app_test_script.py [path_to_video.mp4]")
        print("Or place a .mp4 or .mov video file in the same directory as this script.")
        sys.exit(1)

    test_api(video_path)
