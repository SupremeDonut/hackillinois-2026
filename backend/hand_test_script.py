import modal
import urllib.request
import numpy as np
# Create Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "wget")
    .run_commands("pip install --upgrade pip")
    .pip_install("mediapipe==0.10.9", "opencv-python-headless", "numpy", "protobuf<4")
)

app = modal.App("hand-test-app")


@app.function(image=image)
def test_mediapipe_hands():
    import cv2
    import mediapipe as mp
    import time

    print("Downloading test image...")
    # A clear image of hands from Unsplash
    url = "https://images.unsplash.com/photo-1542362567-b07e54358753?auto=format&fit=crop&q=80&w=500"

    img = None
    try:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    except Exception as e:
        print(
            f"Error downloading or decoding image: {e}. Falling back to blank image.")
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    if img is None:
        print("Failed to load image. Using blank image.")
        img = np.zeros((480, 640, 3), dtype=np.uint8)

    print(f"Image loaded: {img.shape}")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    print("Initializing MediaPipe Hands model...")
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    ) as hands:
        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image and find hand landmarks
        print("Processing image...")
        start_time = time.time()
        results = hands.process(image_rgb)
        elapsed = time.time() - start_time
        print(f"Processing took {elapsed:.3f} seconds.")

        if not results.multi_hand_landmarks:
            print("No hands detected.")
            return 0

        print(
            f"✅ Success! Detected {len(results.multi_hand_landmarks)} hand(s).")

        # Print landmark 0 (WRIST) for each hand detected to verify coordinates
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            print(
                f"  Hand {i+1} Wrist Coord: (x={wrist.x:.3f}, y={wrist.y:.3f}, z={wrist.z:.3f})")

        return len(results.multi_hand_landmarks)


@app.local_entrypoint()
def main():
    print("Running MediaPipe Hands test on Modal...")
    num_hands = test_mediapipe_hands.remote()
    print(f"Test complete. Hands detected: {num_hands}")


if __name__ == '__main__':
    main()
