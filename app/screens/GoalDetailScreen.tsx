import React, { useCallback, useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ScrollView,
    Alert,
    SafeAreaView,
} from 'react-native';
import { useNavigation, useRoute, RouteProp, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import Svg, { Line, Circle, Polyline, Rect, Text as SvgText } from 'react-native-svg';
import { RootStackParamList, Goal } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { loadGoals, deleteGoal } from '../services/goalStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'GoalDetail'>;
type GoalDetailRouteProp = RouteProp<RootStackParamList, 'GoalDetail'>;

// ─── Progression Chart ────────────────────────────────────────────────────────

const CHART_W = 300;
const CHART_H = 140;
const PAD_LEFT = 32;
const PAD_BOTTOM = 28;
const PAD_TOP = 12;
const PAD_RIGHT = 16;
const PLOT_W = CHART_W - PAD_LEFT - PAD_RIGHT;
const PLOT_H = CHART_H - PAD_BOTTOM - PAD_TOP;

function ProgressionChart({ runs }: { runs: Goal['runs'] }) {
    if (runs.length === 0) {
        return (
            <View style={chart.empty}>
                <Text style={chart.emptyText}>No sessions yet — start recording!</Text>
            </View>
        );
    }

    const scores = runs.map((r) => r.score);
    const minScore = Math.max(0, Math.min(...scores) - 10);
    const maxScore = Math.min(100, Math.max(...scores) + 10);
    const range = maxScore - minScore || 1;

    const toX = (i: number) =>
        PAD_LEFT + (runs.length === 1 ? PLOT_W / 2 : (i / (runs.length - 1)) * PLOT_W);
    const toY = (score: number) =>
        PAD_TOP + PLOT_H - ((score - minScore) / range) * PLOT_H;

    const points = runs.map((r, i) => `${toX(i)},${toY(r.score)}`).join(' ');

    // Y-axis labels (3 lines)
    const yLabels = [minScore, Math.round((minScore + maxScore) / 2), maxScore];

    return (
        <Svg width={CHART_W} height={CHART_H}>
            {/* Grid lines + Y labels */}
            {yLabels.map((v) => {
                const y = toY(v);
                return (
                    <React.Fragment key={v}>
                        <Line
                            x1={PAD_LEFT}
                            y1={y}
                            x2={CHART_W - PAD_RIGHT}
                            y2={y}
                            stroke="rgba(255,255,255,0.06)"
                            strokeWidth={1}
                        />
                        <SvgText
                            x={PAD_LEFT - 4}
                            y={y + 4}
                            fontSize={9}
                            fill={Colors.textMuted}
                            textAnchor="end"
                        >
                            {v}
                        </SvgText>
                    </React.Fragment>
                );
            })}

            {/* Axis */}
            <Line
                x1={PAD_LEFT}
                y1={PAD_TOP}
                x2={PAD_LEFT}
                y2={PAD_TOP + PLOT_H}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth={1}
            />
            <Line
                x1={PAD_LEFT}
                y1={PAD_TOP + PLOT_H}
                x2={CHART_W - PAD_RIGHT}
                y2={PAD_TOP + PLOT_H}
                stroke="rgba(255,255,255,0.15)"
                strokeWidth={1}
            />

            {/* Line */}
            <Polyline
                points={points}
                fill="none"
                stroke={Colors.primary}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
            />

            {/* Dots + session labels */}
            {runs.map((r, i) => {
                const x = toX(i);
                const y = toY(r.score);
                const isImproved = r.improvement_delta !== null && (r.improvement_delta ?? 0) > 0;
                const isRegressed = r.improvement_delta !== null && (r.improvement_delta ?? 0) < 0;
                const dotColor = isImproved
                    ? Colors.success
                    : isRegressed
                    ? Colors.error
                    : Colors.primary;
                return (
                    <React.Fragment key={i}>
                        <Circle cx={x} cy={y} r={5} fill={dotColor} />
                        <SvgText
                            x={x}
                            y={PAD_TOP + PLOT_H + 16}
                            fontSize={9}
                            fill={Colors.textMuted}
                            textAnchor="middle"
                        >
                            {i + 1}
                        </SvgText>
                        {/* Score label above dot */}
                        <SvgText
                            x={x}
                            y={y - 8}
                            fontSize={9}
                            fill={dotColor}
                            textAnchor="middle"
                            fontWeight="bold"
                        >
                            {r.score}
                        </SvgText>
                    </React.Fragment>
                );
            })}
        </Svg>
    );
}

// ─── Screen ───────────────────────────────────────────────────────────────────

export default function GoalDetailScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<GoalDetailRouteProp>();
    const { goalId } = route.params;

    const [goal, setGoal] = useState<Goal | null>(null);

    // Reload goal every time screen comes into focus (including after new run)
    useFocusEffect(
        useCallback(() => {
            loadGoals().then((goals) => {
                const found = goals.find((g) => g.id === goalId) ?? null;
                setGoal(found);
            });
        }, [goalId]),
    );

    if (!goal) return null;

    const latestScore = goal.runs.length > 0 ? goal.runs[goal.runs.length - 1].score : null;
    const bestScore = goal.runs.length > 0 ? Math.max(...goal.runs.map((r) => r.score)) : null;
    const totalRuns = goal.runs.length;

    const handleDelete = () => {
        Alert.alert('Delete Goal', `Delete "${goal.name}"? This cannot be undone.`, [
            { text: 'Cancel', style: 'cancel' },
            {
                text: 'Delete',
                style: 'destructive',
                onPress: async () => {
                    await deleteGoal(goalId);
                    navigation.goBack();
                },
            },
        ]);
    };

    const handleRecord = () => {
        navigation.navigate('Recording', {
            activityType: goal.activityType,
            description: goal.name,
            goalId: goal.id,
            voiceId: 's3TPKV1kjDlVtZbl4Ksh', // default: Peter
        });
    };

    return (
        <SafeAreaView style={{ flex: 1, backgroundColor: Colors.background }}>
        <ScrollView
            style={S.screen}
            contentContainerStyle={S.scroll}
            showsVerticalScrollIndicator={false}
        >
            {/* Back + Delete header */}
            <View style={S.header}>
                <TouchableOpacity style={S.backBtn} onPress={() => navigation.goBack()} activeOpacity={0.7}>
                    <Text style={S.backBtnText}>← Back</Text>
                </TouchableOpacity>
                <TouchableOpacity onPress={handleDelete} activeOpacity={0.7}>
                    <Text style={S.deleteText}>Delete</Text>
                </TouchableOpacity>
            </View>

            {/* Goal title */}
            <Text style={S.activityBadge}>{goal.activityType}</Text>
            <Text style={S.goalName}>{goal.name}</Text>

            {/* Stats row */}
            <View style={S.statsRow}>
                <View style={S.stat}>
                    <Text style={S.statValue}>{totalRuns}</Text>
                    <Text style={S.statLabel}>Sessions</Text>
                </View>
                <View style={S.statDivider} />
                <View style={S.stat}>
                    <Text style={S.statValue}>{latestScore ?? '—'}</Text>
                    <Text style={S.statLabel}>Latest Score</Text>
                </View>
                <View style={S.statDivider} />
                <View style={S.stat}>
                    <Text style={[S.statValue, { color: Colors.success }]}>{bestScore ?? '—'}</Text>
                    <Text style={S.statLabel}>Best Score</Text>
                </View>
            </View>

            {/* Progression chart */}
            <View style={S.chartCard}>
                <Text style={S.chartTitle}>Form Score Progression</Text>
                <View style={S.chartWrap}>
                    <ProgressionChart runs={goal.runs} />
                </View>
                {goal.runs.length > 0 && (
                    <Text style={S.chartHint}>
                        {Colors.success && ''}{/* legend dots */}
                        Green dot = improved · Red dot = regressed · Teal = first session
                    </Text>
                )}
            </View>

            {/* Session history */}
            {goal.runs.length > 0 && (
                <View style={S.historyCard}>
                    <Text style={S.chartTitle}>Session History</Text>
                    {[...goal.runs].reverse().map((run, idx) => {
                        const runNum = goal.runs.length - idx;
                        const hasDelta = run.improvement_delta !== null;
                        const positive = (run.improvement_delta ?? 0) > 0;
                        const date = new Date(run.date);
                        const dateStr = `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:${String(date.getMinutes()).padStart(2, '0')}`;
                        return (
                            <View key={idx} style={[S.historyRow, idx > 0 && S.historyRowBorder]}>
                                <View>
                                    <Text style={S.historyRunLabel}>Session {runNum}</Text>
                                    <Text style={S.historyDate}>{dateStr}</Text>
                                </View>
                                <View style={S.historyRight}>
                                    <Text style={S.historyScore}>{run.score}</Text>
                                    {hasDelta && (
                                        <Text style={[S.historyDelta, { color: positive ? Colors.success : Colors.error }]}>
                                            {positive ? '▲' : '▼'} {Math.abs(run.improvement_delta!)}
                                        </Text>
                                    )}
                                </View>
                            </View>
                        );
                    })}
                </View>
            )}

            {/* CTA */}
            <TouchableOpacity style={S.cta} onPress={handleRecord} activeOpacity={0.85}>
                <Text style={S.ctaText}>Start New Session</Text>
            </TouchableOpacity>
        </ScrollView>
        </SafeAreaView>
    );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const chart = StyleSheet.create({
    empty: {
        height: 80,
        alignItems: 'center',
        justifyContent: 'center',
    },
    emptyText: {
        color: Colors.textMuted,
        fontSize: 13,
    },
});

const S = StyleSheet.create({
    screen: { flex: 1, backgroundColor: Colors.background },
    scroll: {
        paddingHorizontal: Spacing.lg,
        paddingTop: 20,
        paddingBottom: 48,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.lg,
    },
    backBtn: {
        paddingVertical: 6,
        paddingRight: 12,
    },
    backBtnText: {
        color: Colors.primary,
        fontSize: 15,
        fontWeight: '600',
    },
    deleteText: {
        color: Colors.error,
        fontSize: 14,
        fontWeight: '600',
    },
    activityBadge: {
        color: Colors.primary,
        fontSize: 11,
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: 1.4,
        marginBottom: 6,
    },
    goalName: {
        color: Colors.text,
        fontSize: 24,
        fontWeight: '800',
        letterSpacing: -0.5,
        marginBottom: Spacing.lg,
    },
    statsRow: {
        flexDirection: 'row',
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        marginBottom: Spacing.lg,
        alignItems: 'center',
        justifyContent: 'space-around',
    },
    stat: { alignItems: 'center', flex: 1 },
    statValue: {
        color: Colors.text,
        fontSize: 26,
        fontWeight: '800',
        letterSpacing: -0.5,
    },
    statLabel: {
        color: Colors.textSecondary,
        fontSize: 11,
        fontWeight: '600',
        textTransform: 'uppercase',
        letterSpacing: 0.8,
        marginTop: 2,
    },
    statDivider: {
        width: 1,
        height: 36,
        backgroundColor: Colors.glassBorder,
    },
    chartCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        marginBottom: Spacing.lg,
    },
    chartTitle: {
        color: Colors.text,
        fontSize: 14,
        fontWeight: '800',
        letterSpacing: -0.2,
        marginBottom: Spacing.sm,
    },
    chartWrap: { alignItems: 'center' },
    chartHint: {
        color: Colors.textMuted,
        fontSize: 11,
        textAlign: 'center',
        marginTop: Spacing.sm,
    },
    historyCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        marginBottom: Spacing.lg,
    },
    historyRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 10,
    },
    historyRowBorder: {
        borderTopWidth: 1,
        borderTopColor: Colors.glassBorder,
    },
    historyRunLabel: {
        color: Colors.text,
        fontSize: 14,
        fontWeight: '600',
    },
    historyDate: {
        color: Colors.textMuted,
        fontSize: 12,
        marginTop: 2,
    },
    historyRight: { alignItems: 'flex-end' },
    historyScore: {
        color: Colors.primary,
        fontSize: 20,
        fontWeight: '800',
    },
    historyDelta: {
        fontSize: 12,
        fontWeight: '700',
        marginTop: 2,
    },
    cta: {
        backgroundColor: Colors.primary,
        borderRadius: Radius.lg,
        paddingVertical: 18,
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.35,
        shadowRadius: 14,
        elevation: 10,
    },
    ctaText: {
        color: Colors.background,
        fontSize: 16,
        fontWeight: '700',
        letterSpacing: 0.2,
    },
});
