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

export type RootStackParamList = {
    Home: undefined;
    Recording: undefined;
    Analyzing: { videoUri: string; activityType: ActivityType; description: string };
    Playback: { videoUri: string; data: AnalysisResponse };
    Complete: { data: AnalysisResponse };
};
