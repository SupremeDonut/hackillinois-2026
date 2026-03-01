export type ActivityType =
    | "basketball_shot"
    | "golf_swing"
    | "tennis_serve"
    | "other";

// ─── Account ─────────────────────────────────────────────────────────────────

export interface Account {
    displayName: string;
    email: string;
    createdAt: string;
}

// ─── Goal / Progression ───────────────────────────────────────────────────────

export interface GoalRun {
    date: string; // ISO timestamp of the session
    score: number; // 0–100 form score
    improvement_delta: number | null;
}

export interface Goal {
    id: string;
    name: string; // user-defined goal description
    activityType: string;
    createdAt: string; // ISO timestamp
    runs: GoalRun[];
}

// ─── Feedback ─────────────────────────────────────────────────────────────────

export interface FeedbackPoint {
    mistake_timestamp_ms: number;
    coaching_script: string;
    visuals: {
        overlay_type: "ANGLE_CORRECTION" | "POSITION_MARKER" | "PATH_TRACE";
        focus_point?: { x: number; y: number };
        vectors?: Array<{
            start: [number, number];
            end: [number, number];
            color: string;
            label?: string;
            is_correction?: boolean;
            body_part?: string;
        }>;
        path_points?: Array<[number, number]>;
        correction_annotations?: Array<{
            pivot: [number, number];
            body_part: string;
            angle_deg: number;
        }>;
    } | null;
    audio_url: string;
    severity?: string;
}

export interface AnalysisResponse {
    status: "success" | "low_confidence" | "error";
    error_message?: string;
    feedback_points: FeedbackPoint[];
    positive_note: string;
    progress_score: number;
    improvement_delta?: number;
}

export type RootStackParamList = {
    Onboarding: undefined;
    Home: undefined;
    Recording: {
        activityType: ActivityType | string;
        description?: string;
        previousData?: AnalysisResponse;
        goalId?: string;
        voiceId: string;
    };
    Analyzing: {
        videoUri: string;
        activityType: ActivityType | string;
        description: string;
        previousData?: AnalysisResponse;
        goalId?: string;
        voiceId: string;
    };
    Playback: {
        videoUri: string;
        data: AnalysisResponse;
        activityType: ActivityType | string;
        goalId?: string;
        voiceId?: string;
    };
    Complete: {
        data: AnalysisResponse;
        activityType: ActivityType | string;
        goalId?: string;
        voiceId?: string;
    };
    GoalDetail: { goalId: string };
    HistoryDetail: { session: HistorySessionParam };
};

// Alias used by nav params (avoids circular import with historyStore)
export interface HistorySessionParam {
    id: string;
    date: string;
    activityType: string;
    score: number;
    improvement_delta: number | null;
    positive_note: string;
    feedback_count: number;
    full_data?: AnalysisResponse;
    linked_goal_id?: string;
}
