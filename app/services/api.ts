/**
 * ==============================================================================
 * 🌐 MODAL API UPLOAD HANDLER (app/services/api.ts)
 * ==============================================================================
 * Stage 4: The one and only network layer for the app.
 *
 * ✅ TO CONNECT THE REAL BACKEND:
 *    Change MODAL_API_URL below from null to your Modal deployment URL.
 *    That's the only change needed. The rest of the flow is already wired up.
 *
 * Fallback Behavior:
 *    If MODAL_API_URL is null OR if the network request fails (crash, timeout,
 *    bad status), the catch block instantly returns the local mock JSON so the
 *    demo always runs flawlessly even without a backend.
 * ==============================================================================
 */

import { AnalysisResponse } from '../types';

// ─────────────────────────────────────────────────────────────────────────────
// 👇 CHANGE THIS ONE LINE WHEN THE BACKEND IS READY
// ─────────────────────────────────────────────────────────────────────────────
const MODAL_API_URL: string | null = 'https://kevinhyang2006--biomechanics-ai-analyze.modal.run';
// const MODAL_API_URL = 'https://YOUR_WORKSPACE--analyze.modal.run';
// ─────────────────────────────────────────────────────────────────────────────

export interface UploadParams {
    videoUri: string;
    activityType: string;
    description: string;
    previousData?: AnalysisResponse; // Forwarded for conversation context
    _useMockRetry?: boolean;         // Dev-only: use retry mock payload when no backend
}

export const uploadVideo = async (params: UploadParams): Promise<AnalysisResponse> => {
    const { videoUri, activityType, description, previousData, _useMockRetry } = params;

    // If no backend URL is configured yet, immediately use mock data
    if (!MODAL_API_URL) {
        console.warn('[API] MODAL_API_URL not set — using mock data.');
        if (_useMockRetry) {
            return require('../data/mock_response_retry.json') as AnalysisResponse;
        }
        return require('../data/mock_response.json') as AnalysisResponse;
    }

    const formData = new FormData();

    // Stream the binary video file (never use Base64 — it OOMs on Android)
    formData.append('video_file', {
        uri: videoUri,
        name: `video_${Date.now()}.mp4`,
        type: 'video/mp4',
    } as any);

    formData.append('activity_type', activityType);
    formData.append('user_description', description);

    // Forward a trimmed version of the previous session as context.
    // Rules:
    //   1. Strip audio_url and visuals — large/binary, irrelevant to the LLM.
    //   2. Only ever send 1 prior session (this naturally caps memory to a 1-attempt buffer;
    //      the previous session passed in here already had its own previous stripped out).
    if (previousData) {
        const trimmedContext = {
            progress_score: previousData.progress_score,
            positive_note: previousData.positive_note,
            improvement_delta: previousData.improvement_delta,
            feedback_points: previousData.feedback_points.map(fp => ({
                mistake_timestamp_ms: fp.mistake_timestamp_ms,
                coaching_script: fp.coaching_script,
                // audio_url and visuals intentionally omitted
            })),
        };
        formData.append('previous_analysis', JSON.stringify(trimmedContext));
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 600_000); // 10 mins timeout

        const response = await fetch(`${MODAL_API_URL}`, {
            method: 'POST',
            body: formData,
            headers: { 'Accept': 'application/json' },
            signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Backend error: ${response.status}`);
        }

        const json = await response.json();
        return json as AnalysisResponse;

    } catch (error) {
        console.error('[API] Request failed, falling back to mock data.', error);
        return require('../data/mock_response.json') as AnalysisResponse;
    }
};

