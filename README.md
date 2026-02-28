# hackillinois-2026

# MotionCoach AI — Technical Specification

> **Context:** 36-hour hackathon project. Prioritize working demo over polish.

---

## 1. Product Overview

**Pitch:** AI coaching for physical hobbies—sports, instruments, dance. Users record a short video, describe their goal, and receive personalized feedback with visual overlays and voice narration.

**Core Loop:**

```
Record (5s max) → Describe intent → AI analyzes → Video pauses at mistake → Overlay + voice feedback → Show progress → Retry
```

**Key Differentiators:**

- Encouraging, low-stress tone (not drill-sergeant)
- Visual "Pause & Draw" feedback (not AI-generated video)
- Progress tracking toward user-defined goals

---

## 2. Architecture

```
┌─────────────────┐         ┌──────────────────────────────────────────────┐
│  React Native   │   HTTP  │                   Modal (A100)                │
│  (Expo + Android)│ ◄─────► │  ┌─────────────────┐    ┌──────────────────┐ │
│                 │         │  │  YOLO26x-Pose   │──▶ │  Qwen3-VL-8B    │ │
│  - expo-camera  │         │  │  (kinematics)   │    │  (Thinking)      │ │
│  - expo-av      │         │  └─────────────────┘    └──────────────────┘ │
│  - react-native-│         │                    ┌──────────────────┐       │
│    svg          │         │                    │   ElevenLabs TTS │       │
└─────────────────┘         │                    └──────────────────┘       │
                            └──────────────────────────────────────────────┘
```

**Stack:**

| Layer | Technology |
|-------|------------|
| Client | React Native (Expo), tested on Android |
| Backend | Modal (Python, serverless **A100 GPU**) |
| Pose Specialist | `yolo26x-pose.pt` via `ultralytics` (NMS-free, January 2026) |
| Vision-Language Logic | `Qwen/Qwen3-VL-8B-Instruct` (Thinking variant) via `transformers` |
| Audio | ElevenLabs API |
| Overlays | react-native-svg (client-side rendering) |

---

## 3. Data Schemas

### 3.1 Request: `POST /analyze`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `video_file` | Binary File | Yes | Send via `multipart/form-data` (Do NOT Base64 encode, it inflates payload size by 33% and causes timeouts) |
| `activity_type` | Enum | Yes | See activity taxonomy below |
| `user_description` | String | Yes | What user wants to improve |
| `goal_id` | UUID or null | No | Links to persistent goal |
| `session_history` | Array | No | Previous sessions for context |

**Activity Type Enum:**

- `basketball_shot`
- `golf_swing`
- `tennis_serve`
- `baseball_pitch`
- `soccer_kick`
- `guitar_strumming`
- `piano_hands`
- `dance_move`
- `other`

**Session History Item Schema:**

| Field | Type |
|-------|------|
| `session_id` | UUID |
| `timestamp` | ISO8601 |
| `progress_score` | Integer (0–100) |
| `main_feedback` | String (summary) |

### 3.2 Response: Analysis Result

| Field | Type | Notes |
|-------|------|-------|
| `status` | Enum | `success`, `low_confidence`, `error` |
| `error_message` | String or null | Only if status is `error` |
| `feedback_points` | Array | See below |
| `positive_note` | String | What user did well |
| `progress_score` | Integer (0–100) | Current attempt rating |
| `improvement_delta` | Integer | Change from last session |

**Feedback Point Object:**

| Field | Type | Notes |
|-------|------|-------|
| `mistake_timestamp_ms` | Integer | Milliseconds into video |
| `coaching_script` | String | Spoken feedback text |
| `visuals` | Object | See below |
| `audio_url` | URL string | ElevenLabs MP3 |

**Visuals Object:**

| Field | Type | Notes |
|-------|------|-------|
| `focus_point` | Object | `{x: 0.0–1.0, y: 0.0–1.0}` |
| `overlay_type` | Enum | `ANGLE_CORRECTION`, `POSITION_MARKER`, `PATH_TRACE` |
| `vectors` | Array | Drawing instructions |

**Vector Schema:**

| Field | Type | Notes |
|-------|------|-------|
| `start` | Array [x, y] | Normalized 0.0–1.0 |
| `end` | Array [x, y] | Normalized 0.0–1.0 |
| `color` | String | `red` (current), `green` (target) |
| `label` | String | `"Current"`, `"Target"` |

### 3.3 Goal Schema (Client-side persistence)

| Field | Type |
|-------|------|
| `goal_id` | UUID |
| `activity_type` | Enum |
| `target_description` | String |
| `created_at` | ISO8601 |
| `sessions` | Array of session IDs |
| `progress_score` | Integer (0–100) |

---

## 4. AI Pipeline (Modal Backend)

### 4.1 The Hybrid YOLO26x + Qwen3-VL Pipeline

**Why this stack:**

- **YOLO26x-Pose** (NMS-free, January 2026): Extracts precise 17-keypoint coordinates frame-by-frame faster than anything before it. Removes guesswork from Qwen — it knows *exactly* where joints are.
- **Qwen3-VL-8B-Thinking**: State-of-the-art open-weight VLM with spatial reasoning. Outperforms Gemini on raw biomechanical logic and runs fully on Modal (no managed API costs or rate limits).

**Pipeline Steps:**

1. **Receive Video:** React Native POSTs the 5-second video to Modal as `multipart/form-data`.
2. **Extract Kinematics:** YOLO26x-Pose processes the video frame-by-frame, extracting `(x, y)` for all 17 keypoints per frame.
3. **Find the Mistake Frame:** Python heuristic on Modal identifies the peak-action frame (e.g., highest joint velocity, lowest wrist Y, etc.).
4. **Grounding Prompt to Qwen3-VL:** The mistake frame image is sent to Qwen3-VL along with the YOLO26 coordinates embedded as text:
   > *"Image attached. User is doing a basketball shot. YOLO26 data: right elbow at (0.45, 0.60), angle 142°. Provide a 1-sentence biomechanical correction."*
5. **Strip `<think>` Tags:** If using the Thinking variant, the model outputs reasoning inside `<think>…</think>` before the answer. **The backend MUST strip this block with regex before sending to ElevenLabs** — otherwise ElevenLabs narrates the internal monologue aloud.
6. **ElevenLabs TTS:** Cleaned `coaching_script` is sent to ElevenLabs. Audio is returned as Base64 (`data:audio/wav;base64,...`) or a presigned URL.
7. **Return Data:** YOLO coordinates (for SVG overlay) + Qwen coaching text + ElevenLabs audio returned to client in the structured JSON schema (see Section 3.2).

**Key Advantage:** Qwen never has to guess where joints are — YOLO26 provides ground-truth coordinates. Qwen only reasons about *why* a position is wrong and what the correction is.

### 4.2 Modal Implementation

```python
import modal

image = (
    modal.Image.debian_slim()
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "ultralytics",          # YOLO26x-Pose
        "transformers>=4.57.0", # Qwen3-VL
        "qwen-vl-utils",
        "opencv-python-headless",
        "torch", "torchvision"
    )
)

app = modal.App("motioncoach-qwen-yolo")

@app.cls(gpu="A100", image=image)
class CoachAI:
    @modal.enter()
    def load_models(self):
        from ultralytics import YOLO
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch
        self.pose_model = YOLO("yolo26x-pose.pt")
        self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    @modal.method()
    def analyze_swing(self, video_bytes: bytes, user_goal: str) -> dict:
        import re, tempfile
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        # 1. Run YOLO26x-Pose (NMS-free, very fast)
        pose_results = self.pose_model(video_path, stream=True)
        # ... extract mistake frame + YOLO coords ...

        # 2. Build grounding prompt for Qwen3-VL
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": mistake_frame},
                {"type": "text", "text": f"User goal: {user_goal}. Right elbow at {mistake_coords}. One-sentence correction:"}
            ]
        }]

        # 3. Generate & strip <think> tags (Thinking variant)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[mistake_frame], return_tensors="pt").to("cuda")
        output = self.qwen_model.generate(**inputs, max_new_tokens=200)
        raw = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        coaching_script = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        return {
            "coaching_script": coaching_script,
            "visuals": {
                "overlay_type": "ANGLE_CORRECTION",
                "focus_point": mistake_coords  # YOLO coords passed straight to React Native
            }
        }
```

### 4.3 ElevenLabs Requirements

- Voice style: "Coach" persona (e.g., 'George' or 'Bella')
- Input: stripped `coaching_script` from Qwen output
- Output: `data:audio/wav;base64,...` (returned inline in the JSON response)
- **CRITICAL:** Strip `<think>` tags before passing to ElevenLabs — Thinking models output internal reasoning that must never be narrated.

### 4.4 Coordinate System (YOLO → SVG)

YOLO26x-Pose outputs keypoint coordinates in **pixel space**. The frontend expects **normalized 0.0–1.0**:

```python
# Normalize YOLO keypoint coords before returning to client
def normalize(kp_x, kp_y, frame_w, frame_h):
    return kp_x / frame_w, kp_y / frame_h
```

The 17 COCO keypoints (0=nose, 5/6=shoulders, 7/8=elbows, 9/10=wrists, etc.) — pass the relevant ones as `vectors` in the response.

---

## 5. Frontend Logic

### 5.1 State Machine

```
IDLE
  │
  ▼ (user taps record)
RECORDING
  │
  ▼ (recording complete, ≤5s enforced)
UPLOADING
  │
  ▼ (upload complete)
ANALYZING
  │
  ▼ (response received)
PLAYBACK
  │
  ▼ (video reaches mistake_timestamp_ms)
PAUSED_FEEDBACK
  │
  ├──▶ (user taps "Continue") ──▶ PLAYBACK (resume)
  │
  ▼ (video ends)
COMPLETE
  │
  ├──▶ (user taps "Try Again") ──▶ RECORDING
  │
  └──▶ (user taps "Done") ──▶ IDLE
```

### 5.2 Playback Synchronization

1. **Load Phase:** Fetch JSON and begin streaming audio URL directly. Preload video into expo-av.
2. **Audio Fixes:** Explicitly configure `Audio.setAudioModeAsync({ playsInSilentModeIOS: true, shouldDuckAndroid: false })`. Mute the video player's background noise so it doesn't drown out the ElevenLabs voice.
3. **Play Phase:** Start video and audio simultaneously
4. **Monitor:** Track `positionMillis` via `onPlaybackStatusUpdate`.
   - **Crucial:** Set `<Video progressUpdateIntervalMillis={50} />`. The default 500ms interval will cause the video to pause on the wrong frame, destroying the visual overlay alignment.
5. **Trigger:** When `positionMillis >= mistake_timestamp_ms`:
   - Pause video
   - Show SVG overlay
   - Audio continues (narrates feedback)
6. **Resume:** User tap dismisses overlay and resumes video. **You must immediately unmount the SVG layer.** Do not try to make static vectors track a moving body.

### 5.3 SVG Coordinate Mapping

Convert normalized coordinates to pixels:

- \( X_{pixel} = X_{gemini} \times BoundingBox_{width} \)
- \( Y_{pixel} = Y_{gemini} \times BoundingBox_{height} \)

**Important (The Aspect Ratio Trap):** Camera sensors rarely match mobile screens (e.g., 4:3 vs 19.5:9).

1. Lock `<CameraView>` and `<Video>` to the exact same aspect ratio (e.g., 16:9).
2. Do not apply math to the entire React Native container. Calculate the exact bounding box of the rendered video within the screen, and apply coordinates *only* to that bounding box to ignore letterboxing bars.

### 5.4 Context Preservation

- Store each session result locally (AsyncStorage or in-memory for demo)
- On next recording, include previous session summaries in request
- Allows AI to say "Last time you were dropping your elbow—let's see if that improved"

---

## 6. Edge Cases

| Scenario | Detection | Response |
|----------|-----------|----------|
| Camera is moving/shaky | User error | UI must aggressively prompt: "Prop phone up or hold perfectly still!" Overlays will fail if the camera pans. |
| No pose detected | YOLO returns no keypoints | Prompt: "We couldn't see your full body. Try standing further back." |
| Video too long | Client-side duration check | Reject before upload: "Please record 5 seconds or less." |
| Network timeout or OOM | Request fails or >15s | Switch to `multipart/form-data`. Show retry option + tips while waiting |
| Qwen returns malformed output | Parse failure | Retry once; fallback to mock |
| Qwen `<think>` not stripped | ElevenLabs narrates reasoning | Backend regex **must** strip `<think>…</think>` before TTS call |
| iOS Silent Switch / Audio Ducking | User device state | Use `playsInSilentModeIOS` and mute video backround track |
| ElevenLabs failure | Audio URL missing or 4xx/5xx | Display text feedback only, skip audio |
| Aspect ratio mismatch | Device info vs video dimensions | Adjust SVG mapping for letterboxing |
| API down during demo | Any backend failure | Load hardcoded mock response |

---

## 7. Screen Inventory

| Screen | Purpose | Key Components |
|--------|---------|----------------|
| Home | Entry point, access goals/record | Goal summary, "Start Recording" button |
| Goals | CRUD for long-term objectives | List view, add/edit modal |
| Recording | Camera capture | Camera view, 5s countdown, activity picker |
| Analyzing | Loading state | Progress indicator, motivational tips |
| Playback | Video + overlay + audio | Video player, SVG layer, controls |
| Complete | Results summary | Progress score, delta, "Try Again" / "Done" |

---

## 8. Implementation Priorities

### P0 — Demo Must Work

- [ ] Camera capture with 5s enforcement, locked aspect ratio, and "Hold Still" UI warning
- [ ] SVG overlay rendering via hardcoded local JSON (Dev 3 unblocked immediately)
- [ ] Modal endpoint: receive `multipart/form-data` video, run YOLO26x-Pose, feed mistake frame + keypoints to Qwen3-VL-8B-Thinking
- [ ] Strip `<think>` tags from Qwen output before passing to ElevenLabs
- [ ] Normalize YOLO keypoint pixel coords to 0.0–1.0 before returning to client
- [ ] Video playback with `progressUpdateIntervalMillis={50}` and background noise muted
- [ ] iOS/Android Audio mode configured to override silent switch
- [ ] ElevenLabs audio returned as Base64 inline (no presigned URL complexity for demo)
- [ ] Mock fallback data for API failures

### P1 — Demo Polish

- [ ] Activity type picker UI
- [ ] Progress score display on complete screen
- [ ] Encouraging tone in AI responses
- [ ] Loading screen with tips

### P2 — If Time Permits

- [ ] Goals tab with persistence
- [ ] Session history storage
- [ ] Context preservation across attempts
- [ ] Improvement delta calculation

---

## 9. Environment Variables

| Key | Location | Purpose |
|-----|----------|---------|
| `ELEVENLABS_API_KEY` | Modal secrets | ElevenLabs TTS |
| `MODAL_API_URL` | React Native `app/services/api.ts` | Backend endpoint (set to `null` during dev to use mock data) |

> [!NOTE]
> No Gemini API key needed — Qwen3-VL runs fully on-device inside the Modal A100 container.

---

## 10. File Structure (Suggested)

```
/app
  App.tsx                # Main Entry Point & Navigation
  /screens
    RecordingScreen.tsx  # Dev 1: Camera and File Upload isolated here
    PlaybackScreen.tsx   # Dev 1: Video looping and AV timing logic
    AnalyzingScreen.tsx  # Dev 1: Calls uploadVideo(), handles mock/real routing
    CompleteScreen.tsx   # Dev 1: Animated progress bar + improvement delta
  /components
    SVGOverlay.tsx       # Dev 3: Isolated drawing engine; takes JSON, draws vectors
  /services
    api.ts               # Dev 1: Multipart upload, mock fallback, context trimming
  /data
    mock_response.json        # Dev 3: First-run mock
    mock_response_retry.json  # Dev 3: Retry mock (with improvement_delta)

/backend
  main.py                # Dev 2: Modal App — YOLO26x → Qwen3-VL → ElevenLabs pipeline
  pose_client.py         # Dev 2: YOLO26x-Pose inference + mistake frame heuristic
  qwen_client.py         # Dev 2: Qwen3-VL prompting, <think> stripping, coord normalization
  elevenlabs_client.py   # Dev 2: TTS call, Base64 encoding
```

---

## 11. Testing the Demo

**Happy Path:**

1. Open app → Tap "Start"
2. Select "Basketball Shot" → Add description
3. Record 5s video
4. Wait for analysis (~5–10s)
5. Video plays → pauses at mistake → overlay appears
6. Voice narrates feedback
7. Tap to continue → video finishes
8. See progress score → Tap "Try Again"

**Failure Path (verify graceful degradation):**

1. Kill network mid-upload → Should show retry
2. Backend returns error → Should load mock data
3. Record 10s video → Should reject before upload
