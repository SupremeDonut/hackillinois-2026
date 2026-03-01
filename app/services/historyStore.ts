import AsyncStorage from '@react-native-async-storage/async-storage';
import { HistorySessionParam } from '../types';

const HISTORY_KEY = '@morphi_session_history';
const MAX_SESSIONS = 50;

// HistorySession is the same shape as HistorySessionParam (used in nav params)
export type HistorySession = HistorySessionParam;

export async function getHistory(): Promise<HistorySession[]> {
    try {
        const raw = await AsyncStorage.getItem(HISTORY_KEY);
        return raw ? (JSON.parse(raw) as HistorySession[]) : [];
    } catch {
        return [];
    }
}

export async function addToHistory(
    session: Omit<HistorySession, 'id'>,
): Promise<void> {
    try {
        const history = await getHistory();
        const newEntry: HistorySession = { ...session, id: Date.now().toString() };
        const updated = [newEntry, ...history].slice(0, MAX_SESSIONS);
        await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
    } catch (e) {
        console.warn('[History] Failed to save session:', e);
    }
}

export async function updateHistorySession(
    id: string,
    patch: Partial<HistorySession>,
): Promise<void> {
    try {
        const history = await getHistory();
        const updated = history.map(s => (s.id === id ? { ...s, ...patch } : s));
        await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(updated));
    } catch (e) {
        console.warn('[History] Failed to update session:', e);
    }
}

export async function clearHistory(): Promise<void> {
    await AsyncStorage.removeItem(HISTORY_KEY);
}
