import React, { useState, useCallback } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, ScrollView, Alert,
} from 'react-native';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Colors, Spacing, Radius } from '../styles/theme';
import { getHistory, clearHistory, HistorySession } from '../services/historyStore';
import { RootStackParamList } from '../types';

type NavProp = NativeStackNavigationProp<RootStackParamList>;

function formatDate(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatTime(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

const GRADE_COLORS: Record<string, string> = {
    Elite: '#00E5A0',
    Good: '#90EDB0',
    Progress: '#F5C84A',
    'Keep Going': '#F07070',
};

function gradeFor(score: number): string {
    if (score >= 90) return 'Elite';
    if (score >= 75) return 'Good';
    if (score >= 60) return 'Progress';
    return 'Keep Going';
}

export default function HistoryTab() {
    const navigation = useNavigation<NavProp>();
    const [sessions, setSessions] = useState<HistorySession[]>([]);

    useFocusEffect(
        useCallback(() => {
            getHistory().then(setSessions);
        }, []),
    );

    const handleClear = () => {
        Alert.alert(
            'Clear History',
            'Remove all saved sessions? This cannot be undone.',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Clear All',
                    style: 'destructive',
                    onPress: async () => {
                        await clearHistory();
                        setSessions([]);
                    },
                },
            ],
        );
    };

    return (
        <ScrollView
            style={S.screen}
            contentContainerStyle={S.scroll}
            showsVerticalScrollIndicator={false}
        >
            {/* Header */}
            <View style={S.header}>
                <Text style={S.title}>History</Text>
                {sessions.length > 0 && (
                    <TouchableOpacity onPress={handleClear} activeOpacity={0.7}>
                        <Text style={S.clearBtn}>Clear</Text>
                    </TouchableOpacity>
                )}
            </View>

            {/* Empty state */}
            {sessions.length === 0 && (
                <View style={S.emptyContainer}>
                    <Text style={S.emptyTitle}>No sessions yet</Text>
                    <Text style={S.emptyBody}>
                        Complete a recording session and it will appear here.
                    </Text>
                </View>
            )}

            {/* Session list */}
            {sessions.map(session => {
                const grade = gradeFor(session.score);
                const gradeColor = GRADE_COLORS[grade];
                const hasDelta = session.improvement_delta !== null;
                const positive = (session.improvement_delta ?? 0) >= 0;
                return (
                    <TouchableOpacity
                        key={session.id}
                        style={S.card}
                        onPress={() => navigation.navigate('HistoryDetail', { session })}
                        activeOpacity={0.75}
                    >
                        {/* Top row */}
                        <View style={S.cardTop}>
                            <View style={{ flex: 1 }}>
                                <Text style={S.activityLabel}>{session.activityType}</Text>
                                <Text style={S.dateText}>
                                    {formatDate(session.date)} · {formatTime(session.date)}
                                </Text>
                            </View>
                            <View style={{ alignItems: 'flex-end' }}>
                                <Text style={S.scoreNum}>{session.score}</Text>
                                <View style={[S.gradePill, { borderColor: gradeColor + '77', backgroundColor: gradeColor + '22' }]}>
                                    <Text style={[S.gradeText, { color: gradeColor }]}>{grade}</Text>
                                </View>
                            </View>
                        </View>

                        {/* Bottom row */}
                        <View style={S.cardBottom}>
                            <Text style={S.metaText}>
                                {session.feedback_count} issue{session.feedback_count !== 1 ? 's' : ''} flagged
                            </Text>
                            {hasDelta && (
                                <Text style={[S.deltaText, { color: positive ? Colors.success : Colors.error }]}>
                                    {positive ? '▲' : '▼'} {Math.abs(session.improvement_delta!)} pts
                                </Text>
                            )}
                        </View>

                        {session.positive_note ? (
                            <Text style={S.note} numberOfLines={2}>{session.positive_note}</Text>
                        ) : null}
                    </TouchableOpacity>
                );
            })}
        </ScrollView>
    );
}

const S = StyleSheet.create({
    screen: { flex: 1, backgroundColor: Colors.background },
    scroll: {
        paddingHorizontal: Spacing.lg,
        paddingTop: 60,
        paddingBottom: 100,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.lg,
    },
    title: {
        fontSize: 28,
        fontWeight: '800',
        color: Colors.text,
        letterSpacing: -0.8,
    },
    clearBtn: {
        color: Colors.error,
        fontSize: 14,
        fontWeight: '600',
    },
    emptyContainer: {
        marginTop: Spacing.xxl,
        alignItems: 'center',
        paddingHorizontal: Spacing.lg,
    },
    emptyTitle: {
        fontSize: 18,
        fontWeight: '800',
        color: Colors.text,
        marginBottom: Spacing.sm,
    },
    emptyBody: {
        fontSize: 14,
        color: Colors.textMuted,
        textAlign: 'center',
        lineHeight: 22,
    },
    card: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        borderLeftWidth: 3,
        borderLeftColor: Colors.primary,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
        overflow: 'hidden',
    },
    cardTop: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        marginBottom: Spacing.sm,
    },
    activityLabel: {
        color: Colors.primary,
        fontSize: 10,
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: 1.2,
        marginBottom: 3,
    },
    dateText: {
        color: Colors.textMuted,
        fontSize: 12,
    },
    scoreNum: {
        color: Colors.text,
        fontSize: 28,
        fontWeight: '800',
        lineHeight: 30,
        letterSpacing: -1,
        marginBottom: 4,
    },
    gradePill: {
        borderRadius: Radius.full,
        borderWidth: 1,
        paddingHorizontal: 10,
        paddingVertical: 3,
    },
    gradeText: {
        fontSize: 10,
        fontWeight: '800',
        letterSpacing: 0.8,
        textTransform: 'uppercase',
    },
    cardBottom: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    metaText: {
        color: Colors.textSecondary,
        fontSize: 12,
    },
    deltaText: {
        fontSize: 12,
        fontWeight: '700',
    },
    note: {
        marginTop: Spacing.sm,
        fontSize: 13,
        color: Colors.textSecondary,
        lineHeight: 18,
        fontStyle: 'italic',
    },
});
