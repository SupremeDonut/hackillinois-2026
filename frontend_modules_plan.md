# Frontend Modules Implementation Plan (Developer 1)

**Role:** Developer 1
**Primary Focus:** Camera capture, AV synchronization, API networking, and core screen flows.
**Tech Stack:** React Native (Expo), `expo-camera`, `expo-av`, Native `StyleSheet`.

---

## Development Stages & Testing Plan

To avoid a "big bang" integration where nothing works, we will build and test the app in 4 distinct phases. Do not move to the next phase until the current one is verified on a physical device.

### Stage 1: The UI Shell & Static Architecture

**Goal:** Build the navigation skeleton and verify you can move between all 5 screens.

- **Tasks:**
  1. Initialize Expo router / `@react-navigation/native-stack`.
  2. Create all 5 screen files (`HomeScreen.tsx`, `RecordingScreen.tsx`, etc.) with basic "Hello World" `<Text>` placeholders.
  3. Create `styles/theme.ts` and `styles/globalStyles.ts`.
  4. Write mock navigation buttons to force transitions: Home -> Recording -> Analyzing -> Playback -> Complete.
- **Testing criteria:** You can tap through the entire user flow. State (like a dummy string) can be passed from the Home screen all the way to the Complete screen.

### Stage 2: Camera Capture & Local Saving (Recording Screen)

**Goal:** Prove we can capture exactly 5 seconds of `16:9` video and save it to the device's local cache.

- **Tasks:**
  1. Add `expo-camera` to `RecordingScreen.tsx`.
  2. Add permission handling (Microphone/Camera).
  3. Implement the Start/Stop recording button.
  4. Force a 16:9 aspect ratio using `StyleSheet`.
  5. Add the 5.0 second auto-timeout logic.
- **Testing criteria:** Upon tapping "Start", the camera records for 5s, stops, and `console.log`s a valid local `.mp4` `uri`.

### Stage 3: The Playback Engine (Playback Screen)

**Goal:** Prove we can play a video, pause it precisely, and play audio simultaneously—**without the backend**.

- **Tasks:**
  1. Add `expo-av` Video and Audio components to `PlaybackScreen.tsx`.
  2. Create `data/mock_response.json` (hardcode a schema-compliant response, e.g., `mistake_timestamp_ms: 2500`).
  3. Set `progressUpdateIntervalMillis={50}` on the Video player.
  4. Build the "Sync Engine": When the video hits `2500ms`, pause the video and play a local/dummy audio file.
- **Testing criteria:** The user flow is now: Home -> Record a 5s video -> (Skip API) -> Playback Screen loads the video you just recorded -> Video plays -> Pauses precisely at 2.5s while an audio file plays.

### Stage 4: Network & Backend Integration (Analyzing Screen & API)

**Goal:** Connect the real camera footage to the real Python backend.

- **Tasks:**
  1. Build `services/api.ts` `analyzeVideo` function using `multipart/form-data`.
  2. In `AnalyzingScreen.tsx`, call the API with the recorded video URI.
  3. Map the returned JSON directly into the Playback screen from Stage 3.
  4. Implement HTTP error handling / timeout fallbacks (load the mock JSON if API fails).
- **Testing criteria:**
  1. Happy path: Record video -> Upload takes ~10s -> Real AI audio and real mistake timestamps drive the Playback sync.
  2. Failure path: Turn off WiFi -> Record video -> App gracefully falls back to mock data or shows a retry button.

---

## 1. Project Setup & Architecture

### 1.1 Directory Structure

```
/app
├── App.tsx                 # Navigation container & global state providers
├── screens/
│   ├── HomeScreen.tsx      # Entry point
│   ├── RecordingScreen.tsx # Stage 2: Camera
│   ├── AnalyzingScreen.tsx # Stage 4: Network
│   ├── PlaybackScreen.tsx  # Stage 3: AV Sync
│   └── CompleteScreen.tsx  # Stage 1: UI Shell
├── components/
│   ├── CameraCapture.tsx   
│   ├── AnalysisLoader.tsx  
│   ├── VideoPlayer.tsx     
│   └── SVGOverlay.tsx      # Renders visuals (Handled by Dev 3)
├── services/
│   ├── api.ts              # Stage 4: Network logic
│   └── storage.ts          
├── styles/
│   ├── theme.ts            
│   └── globalStyles.ts     
└── types/
    └── index.ts            
```

### 1.2 Global Types (`types/index.ts`)

Must exactly match the backend schemas from the technical specification.

```typescript
export type ActivityType = 'basketball_shot' | 'golf_swing' | 'tennis_serve' | 'other';

export interface AnalysisResponse {
  status: 'success' | 'low_confidence' | 'error';
  error_message?: string;
  analysis: {
    mistake_timestamp_ms: number;
    coaching_script: string;
    positive_note: string;
    progress_score: number;
    improvement_delta?: number;
  };
  visuals: {
    focus_point: { x: number; y: number };
    overlay_type: 'ANGLE_CORRECTION' | 'POSITION_MARKER';
    vectors: Array<{
      start: [number, number];
      end: [number, number];
      color: string;
      label: string;
    }>;
  };
  audio_url: string;
}
```

---

## 2. Core Modules Detailed Logic

### 2.1 API Service (`services/api.ts`) -> Stage 4

**Responsibilities:**
Upload the raw `.mp4` file and metadata to Modal.

**Key Logic:**

```typescript
export const analyzeVideo = async (uri: string, activityType: string, description: string): Promise<AnalysisResponse> => {
  const formData = new FormData();
  formData.append('video_file', {
    uri: uri,
    name: 'upload.mp4',
    type: 'video/mp4',
  } as any);
  formData.append('activity_type', activityType);
  formData.append('user_description', description);

  // CRITICAL: Fetch configuration
  // - Method: POST
  // - Headers: Content-Type: multipart/form-data
  // - Timeout logic: Implement an AbortController for 15s timeout
};
```

### 2.2 Playback Synchronization Screen (`screens/PlaybackScreen.tsx`) -> Stage 3

**Responsibilities:**
Flawlessly sync video playback, AI audio narration, and SVG overlay rendering.

**Component State:**

```typescript
const [isPausedForFeedback, setIsPausedForFeedback] = useState(false);
const [hasShownFeedback, setHasShownFeedback] = useState(false);
const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
```

**Key Logic / Order of Operations:**

1. **Audio Prep (onMount):**
   - `Audio.setAudioModeAsync({ playsInSilentModeIOS: true, shouldDuckAndroid: false })`
   - Preload the ElevenLabs `data.audio_url` using `Audio.Sound.createAsync`.
2. **Video Config:**
   - `<Video source={{ uri: videoUri }} progressUpdateIntervalMillis={50} isMuted={true} />`
   - Capture exact video rendering dimensions via `onLayout` for the SVG bounding box.
3. **The Sync Engine:**

   ```typescript
   const handlePlaybackStatusUpdate = (status: AVPlaybackStatus) => {
     if (!status.isLoaded) return;
     
     // Trigger Point
     if (!hasShownFeedback && status.positionMillis >= data.analysis.mistake_timestamp_ms) {
       videoRef.current.pauseAsync();
       setIsPausedForFeedback(true);
       setHasShownFeedback(true);
       audioRef.current.playAsync(); // Start playing the AI voice
     }

     // End of video
     if (status.didJustFinish) {
       navigation.replace('Complete', { data });
     }
   };
   ```

4. **Resuming:** When the user taps the screen to continue, hide the SVG overlay (set `isPausedForFeedback=false`) and call `videoRef.current.playAsync()`.

---

## 3. Styling Specs (Native `StyleSheet`)

All components must use `StyleSheet.create`. No inline styles for layout.

```typescript
// styles/theme.ts
export const Colors = {
  background: '#121212', // Pure dark mode
  surface: '#1E1E1E',
  primary: '#4CAF50', // Encouraging green
  text: '#FFFFFF',
  textSecondary: '#AAAAAA',
  error: '#FF5252',
  vectorCurrent: '#FF3B30', // SVG Red
  vectorTarget: '#34C759',  // SVG Green
};

// styles/globalStyles.ts
export const S = StyleSheet.create({
  fullScreen: { 
    flex: 1, 
    backgroundColor: Colors.background 
  },
  centerContent: { 
    flex: 1, 
    justifyContent: 'center', 
    alignItems: 'center' 
  },
  aspectRatio169: {
    width: '100%',
    aspectRatio: 9 / 16, // Assuming portrait orientation for the video
    backgroundColor: '#000'
  },
  primaryButton: {
    backgroundColor: Colors.primary,
    paddingVertical: 16,
    paddingHorizontal: 32,
    borderRadius: 12,
    alignItems: 'center',
  },
  overlayContainer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 10,
    // Must be positioned perfectly over the video player boundaries to match Gemini coordinates
  }
});
```
