export type ActivityType = 'basketball_shot' | 'golf_swing' | 'tennis_serve' | 'other';

// ─── Account ─────────────────────────────────────────────────────────────────

export interface Account {
    displayName: string;
    email: string;
    createdAt: string;
}

// ─── Goal / Progression ───────────────────────────────────────────────────────

export interface GoalRun {
    date: string;          // ISO timestamp of the session
    score: number;         // 0–100 form score
    improvement_delta: number | null;
}

export interface Goal {
    id: string;
    name: string;          // user-defined goal description
    activityType: string;
    createdAt: string;     // ISO timestamp
    runs: GoalRun[];
}

// ─── Feedback ─────────────────────────────────────────────────────────────────

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
    } | null;
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
    Onboarding: undefined;
    Home: undefined;
    Recording: { activityType: ActivityType | string; description?: string; previousData?: AnalysisResponse; goalId?: string };
    Analyzing: { videoUri: string; activityType: ActivityType | string; description: string; previousData?: AnalysisResponse; goalId?: string };
    Playback: { videoUri: string; data: AnalysisResponse; activityType: ActivityType | string; goalId?: string };
    Complete: { data: AnalysisResponse; activityType: ActivityType | string; goalId?: string };
    GoalDetail: { goalId: string };
};
