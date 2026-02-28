# Backend API Contract

## Overview

The React Native frontend sends a video recording to a single endpoint and expects a structured JSON analysis in return. Everything the frontend needs to display coaching feedback — timing, text, audio, and visual overlays — comes from this one response.

---

## Endpoint

```
POST /analyze
Content-Type: multipart/form-data
```

**To activate the live backend:** set `MODAL_API_URL` in `app/services/api.ts` to your deployed Modal URL.  
**If the endpoint is unreachable or returns an error:** the frontend automatically falls back to mock data — no crash.

---

## Request Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `video_file` | Binary file (`.mp4`) | ✅ | The recorded video. Sent as raw binary stream — do NOT expect Base64. |
| `activity_type` | String | ✅ | What the user is practicing, e.g. `"Basketball Shot"`, `"Badminton Smash"` |
| `user_description` | String | ✅ | Supplementary free-text context from the user |
| `previous_analysis` | JSON string | ❌ | Trimmed context from the immediately prior session (if "Try Again" was tapped). Always at most 1 session. See **Session Memory** below. |

---

## Session Memory

The frontend implements a **1-attempt rolling buffer**. Only the immediately preceding session is ever forwarded — the model has no access to anything older.

`previous_analysis` is a trimmed JSON object (audio and visual data are stripped):

```json
{
  "progress_score": 85,
  "positive_note": "Your footwork looks really solid today.",
  "improvement_delta": null,
  "feedback_points": [
    { "mistake_timestamp_ms": 1500, "coaching_script": "Your elbow dropped a bit early here." }
  ]
}
```

Use this to:

- Reference what was already corrected ("Last time we noted your elbow — let's look at your wrist now")
- Generate an accurate `improvement_delta` by comparing the new `progress_score` with the prior one
- Avoid repeating the same feedback if the user has already improved on that point

---

## Response

Return `Content-Type: application/json`.

```json
{
  "status": "success",
  "positive_note": "Your footwork looks really solid today.",
  "progress_score": 82,
  "improvement_delta": 5,
  "feedback_points": [
    {
      "mistake_timestamp_ms": 1500,
      "coaching_script": "Your elbow dropped a bit early here.",
      "audio_url": "data:audio/wav;base64,UklGRqQMAABXQVZF...",
      "visuals": {
        "overlay_type": "POSITION_MARKER",
        "focus_point": { "x": 0.45, "y": 0.35 },
        "vectors": [
          {
            "start": [0.4, 0.3],
            "end": [0.6, 0.5],
            "color": "#FF3B30",
            "label": "Drop Point"
          }
        ],
        "path_points": []
      }
    }
  ]
}
```

---

## Field Reference

### Top Level

| Field | Type | Required | Description |
|---|---|---|---|
| `status` | `"success"` \| `"low_confidence"` \| `"error"` | ✅ | Result status |
| `positive_note` | String | ✅ | One encouraging sentence shown on the results screen |
| `progress_score` | Integer `0–100` | ✅ | Overall form score, shown as an animated progress bar |
| `improvement_delta` | Integer | ❌ | Score change vs. previous session (e.g. `+5`) |
| `error_message` | String | ❌ | Human-readable error, only if `status` is `"error"` |
| `feedback_points` | Array | ✅ | Ordered list of coaching moments. May be empty `[]`. |

### `feedback_points[]`

| Field | Type | Required | Description |
|---|---|---|---|
| `mistake_timestamp_ms` | Integer | ✅ | Millisecond into the video where playback pauses |
| `coaching_script` | String | ✅ | Text shown to the user explaining the correction |
| `audio_url` | String | ✅ | Base64-encoded audio: `"data:audio/wav;base64,..."`. This is read aloud while the video is paused. |
| `visuals` | Object | ✅ | SVG overlay data drawn on top of the paused frame |

### `visuals`

| Field | Type | Required | Description |
|---|---|---|---|
| `overlay_type` | `"ANGLE_CORRECTION"` \| `"POSITION_MARKER"` \| `"PATH_TRACE"` | ✅ | Controls which SVG drawing mode is used |
| `focus_point` | `{ x, y }` | ❌ | Crosshair drawn at this relative position (used in `POSITION_MARKER`) |
| `vectors` | Array of vector objects | ❌ | Lines drawn on screen (used in `ANGLE_CORRECTION` and `POSITION_MARKER`) |
| `path_points` | Array of `[x, y]` pairs | ❌ | A traced arc or curve (used in `PATH_TRACE`) |

### Coordinates — **CRITICAL**

> All coordinates are **relative values between `0.0` and `1.0`**, not pixels.  
> `[0.0, 0.0]` = top-left corner of the video frame.  
> `[1.0, 1.0]` = bottom-right corner of the video frame.  
>
> ✅ `"start": [0.45, 0.3]`  
> ❌ `"start": [324, 210]`  
>
> The frontend SVG engine maps these to absolute pixels at render time based on the actual screen dimensions.

### Vector Object

```json
{
  "start": [0.4, 0.3],
  "end": [0.6, 0.5],
  "color": "#FF3B30",
  "label": "Elbow"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `start` | `[x, y]` | ✅ | Start point (relative `0.0–1.0`) |
| `end` | `[x, y]` | ✅ | End point (relative `0.0–1.0`) |
| `color` | Hex string | ✅ | Line color, e.g. `"#FF3B30"` (red) or `"#34C759"` (green) |
| `label` | String | ❌ | Small text label rendered near the end point |

### Audio

The `audio_url` field must be a data URI containing a **Base64-encoded WAV or MP3**:

```
data:audio/wav;base64,UklGRqQMAABXQVZFZm10IB...
```

- WAV (any sample rate, mono or stereo) is preferred for reliability.
- MP3 is also accepted.
- Do **not** send a plain URL — the frontend plays this audio offline from the data URI.

---

## `overlay_type` Behavior

| Value | What is drawn |
|---|---|
| `POSITION_MARKER` | A focus crosshair at `focus_point` + any vectors |
| `ANGLE_CORRECTION` | Vectors + an arc between the first two vectors showing the angle |
| `PATH_TRACE` | The `path_points` array rendered as a continuous curved line + any vectors |

---

## Error Response

If analysis fails, return:

```json
{
  "status": "error",
  "error_message": "Could not detect a human pose in the video.",
  "positive_note": "",
  "progress_score": 0,
  "feedback_points": []
}
```

The frontend will display this gracefully without crashing.
