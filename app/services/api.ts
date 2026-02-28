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
const MODAL_API_URL: string | null = null;
// const MODAL_API_URL = 'https://YOUR_WORKSPACE--analyze.modal.run';
// ─────────────────────────────────────────────────────────────────────────────

export interface UploadParams {
    videoUri: string;
    activityType: string;
    description: string;
    previousData?: AnalysisResponse; // Forwarded for conversation context
}

export const uploadVideo = async (params: UploadParams): Promise<AnalysisResponse> => {
    const { videoUri, activityType, description, previousData } = params;

    // If no backend URL is configured yet, immediately use mock data
    if (!MODAL_API_URL) {
        console.warn('[API] MODAL_API_URL not set — using mock data.');
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

    // Optional: pass the prior session's analysis so the LLM can reference it
    if (previousData) {
        formData.append('previous_analysis', JSON.stringify(previousData));
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 20_000); // 20s timeout

        const response = await fetch(`${MODAL_API_URL}/analyze`, {
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

