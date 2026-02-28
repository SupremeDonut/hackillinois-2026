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
┌─────────────────┐         ┌─────────────────────────────────┐
│  React Native   │   HTTP  │            Modal                │
│  (Expo + Android)│ ◄─────► │  ┌─────────┐    ┌───────────┐  │
│                 │         │  │ Gemini  │    │ ElevenLabs│  │
│  - expo-camera  │         │  │ 3.1     │    │ TTS       │  │
│  - expo-av      │         │  │ Flash   │    │           │  │
│  - react-native-│         │  └─────────┘    └───────────┘  │
│    svg          │         │                                 │
└─────────────────┘         └─────────────────────────────────┘
```

**Stack:**
| Layer | Technology |
|-------|------------|
| Client | React Native (Expo), tested on Android |
| Backend | Modal (Python, serverless GPU) |
| Vision | Gemini 3.1 Flash (native video input) |
| Audio | ElevenLabs API |
| Overlays | react-native-svg (client-side rendering) |

---

## 3. Data Schemas

### 3.1 Request: `POST /analyze`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `video_file` | Base64 string | Yes | Max 5 seconds, 720p |
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
| `analysis` | Object | See below |
| `visuals` | Object | See below |
| `audio_url` | URL string | ElevenLabs MP3 |

**Analysis Object:**
| Field | Type | Notes |
|-------|------|-------|
| `mistake_timestamp_ms` | Integer | Milliseconds into video |
| `coaching_script` | String | Spoken feedback text |
| `positive_note` | String | What user did well |
| `progress_score` | Integer (0–100) | Current attempt rating |
| `improvement_delta` | Integer | Change from last session |
| `technical_stats` | Object | `observed_angle`, `target_angle` (optional) |

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

### 4.1 Gemini 3.1 Flash Configuration

**Model:** `gemini-3.1-flash` (prioritize for speed) or `gemini-3.1-pro` (if higher accuracy needed)

**Why Gemini 3.1:**
- Native multimodal video understanding (no frame extraction needed)
- Direct .mp4 input via API
- Fast inference for real-time coaching feedback

**System Prompt Must Include:**
1. Role: Supportive biomechanics coach for hobbyists
2. Tone rules:
   - Start with something positive
   - Frame corrections as suggestions ("try this") not criticisms
   - End with encouragement
3. Technical requirements:
   - Identify the single most impactful correction
   - Return exact millisecond of error visibility
   - Return X/Y coordinates as floats (0.0–1.0) relative to frame
   - Output strict JSON only

**Context Injection:**
- Include `activity_type` in prompt
- Include `user_description` (their goal)
- Include summarized `session_history` if available (what they were working on previously)

**Coordinate System:**
- Origin: Top-left of video frame
- X: 0.0 (left) to 1.0 (right)
- Y: 0.0 (top) to 1.0 (bottom)

### 4.2 ElevenLabs Requirements

- Voice style: "Coach" persona (e.g., 'George' or 'Bella')
- High stability setting for clarity
- Input: `coaching_script` from Gemini response
- Output: MP3 URL or base64

### 4.3 Pipeline Sequence

1. Receive video + metadata from client
2. Call Gemini 3.1 Flash with video + constructed prompt
3. Parse and validate JSON response
4. Call ElevenLabs with `coaching_script`
5. Return combined response to client

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

1. **Load Phase:** Fetch JSON and MP3, preload both into expo-av
2. **Play Phase:** Start video and audio simultaneously
3. **Monitor:** Track `positionMillis` via `onPlaybackStatusUpdate`
4. **Trigger:** When `positionMillis >= mistake_timestamp_ms`:
   - Pause video
   - Show SVG overlay
   - Audio continues (narrates feedback)
5. **Resume:** User tap dismisses overlay and resumes video

### 5.3 SVG Coordinate Mapping

Convert normalized coordinates to pixels:
- \( X_{pixel} = X_{gemini} \times Container_{width} \)
- \( Y_{pixel} = Y_{gemini} \times Container_{height} \)

**Important:** Account for video `resizeMode`. If using `contain`, calculate letterboxing offset.

### 5.4 Context Preservation

- Store each session result locally (AsyncStorage or in-memory for demo)
- On next recording, include previous session summaries in request
- Allows AI to say "Last time you were dropping your elbow—let's see if that improved"

---

## 6. Edge Cases

| Scenario | Detection | Response |
|----------|-----------|----------|
| No movement detected | Gemini returns `low_confidence` status | Prompt: "We couldn't see your full body. Try standing further back." |
| Video too long | Client-side duration check | Reject before upload: "Please record 5 seconds or less." |
| Network timeout | Request fails or >15s | Show retry option + tips while waiting |
| Gemini returns malformed JSON | JSON parse failure | Retry once with stricter prompt; fallback to mock |
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
- [ ] Camera capture with 5s enforcement
- [ ] Modal endpoint: receive video, call Gemini 3.1 Flash, return JSON
- [ ] Video playback with timestamp-triggered pause
- [ ] SVG overlay rendering from vector data
- [ ] ElevenLabs audio playback
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
| `GEMINI_API_KEY` | Modal secrets | Gemini 3.1 API access |
| `ELEVENLABS_API_KEY` | Modal secrets | ElevenLabs API access |
| `MODAL_API_URL` | React Native .env | Backend endpoint |

---

## 10. File Structure (Suggested)

```
/app
  /screens
    HomeScreen.tsx
    RecordingScreen.tsx
    PlaybackScreen.tsx
    GoalsScreen.tsx
  /components
    SVGOverlay.tsx
    ActivityPicker.tsx
    ProgressCard.tsx
  /hooks
    useCoachingSession.ts
    useVideoPlayer.ts
  /services
    api.ts
    storage.ts
  /types
    schemas.ts

/backend
  main.py              # Modal app entry
  gemini_client.py     # Gemini 3.1 prompt logic
  elevenlabs_client.py # TTS logic
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