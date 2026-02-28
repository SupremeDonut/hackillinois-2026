import AsyncStorage from '@react-native-async-storage/async-storage';
import { Goal, GoalRun } from '../types';

const STORAGE_KEY = '@motioncoach_goals';

// ─── Helpers ──────────────────────────────────────────────────────────────────

function generateId(): string {
    return `goal_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
}

// ─── CRUD ─────────────────────────────────────────────────────────────────────

export async function loadGoals(): Promise<Goal[]> {
    try {
        const raw = await AsyncStorage.getItem(STORAGE_KEY);
        return raw ? (JSON.parse(raw) as Goal[]) : [];
    } catch {
        return [];
    }
}

export async function saveGoals(goals: Goal[]): Promise<void> {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(goals));
}

export async function createGoal(name: string, activityType: string): Promise<Goal> {
    const goals = await loadGoals();
    const newGoal: Goal = {
        id: generateId(),
        name: name.trim(),
        activityType,
        createdAt: new Date().toISOString(),
        runs: [],
    };
    await saveGoals([...goals, newGoal]);
    return newGoal;
}

export async function deleteGoal(goalId: string): Promise<void> {
    const goals = await loadGoals();
    await saveGoals(goals.filter((g) => g.id !== goalId));
}

export async function addRunToGoal(goalId: string, run: GoalRun): Promise<Goal | null> {
    const goals = await loadGoals();
    const idx = goals.findIndex((g) => g.id === goalId);
    if (idx === -1) return null;
    goals[idx] = { ...goals[idx], runs: [...goals[idx].runs, run] };
    await saveGoals(goals);
    return goals[idx];
}

export async function getGoal(goalId: string): Promise<Goal | null> {
    const goals = await loadGoals();
    return goals.find((g) => g.id === goalId) ?? null;
}
