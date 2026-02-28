export type ActivityType = 'basketball_shot' | 'golf_swing' | 'tennis_serve' | 'other';

export interface FeedbackPoint {
    mistake_timestamp_ms: number;
    coaching_script: string;
    visuals: {
        overlay_type: 'ANGLE_CORRECTION' | 'POSITION_MARKER' | 'PATH_TRACE';
        focus_point?: { x: number; y: number };
        vectors?: Array<{
            start: [number, number];
            end: [number, number];
            color: string;
            label?: string;
        }>;
        path_points?: Array<[number, number]>;
    };
    audio_url: string;
}

export interface AnalysisResponse {
    status: 'success' | 'low_confidence' | 'error';
    error_message?: string;
    feedback_points: FeedbackPoint[];
    positive_note: string;
    progress_score: number;
    improvement_delta?: number;
}

export type RootStackParamList = {
    Home: undefined;
    Recording: { activityType: ActivityType | string; previousData?: AnalysisResponse };
    Analyzing: { videoUri: string; activityType: ActivityType | string; description: string; previousData?: AnalysisResponse };
    Playback: { videoUri: string; data: AnalysisResponse; activityType: ActivityType | string };
    Complete: { data: AnalysisResponse; activityType: ActivityType | string };
};
