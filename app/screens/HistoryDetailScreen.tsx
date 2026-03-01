import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, ScrollView,
    ActivityIndicator, Modal, FlatList, Alert,
} from 'react-native';
import { useNavigation, useRoute, RouteProp, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Audio } from 'expo-av';
import { RootStackParamList } from '../types';
import { FeedbackPoint } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { updateHistorySession } from '../services/historyStore';
import { loadGoals, addRunToGoal } from '../services/goalStore';
import type { Goal } from '../types';

type NavProp = NativeStackNavigationProp<RootStackParamList, 'HistoryDetail'>;
type RoutePropType = RouteProp<RootStackParamList, 'HistoryDetail'>;

const BAR_WIDTH = 280;
type SeverityTab = 'major' | 'intermediate' | 'minor';

const GRADE_COLORS: Record<string, string> = {
    Elite: '#00E5A0',
    Good: '#90EDB0',
    Progress: '#F5C84A',
    'Keep Going': '#F07070',
};

function formatDate(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
}
function formatTime(iso: string): string {
    const d = new Date(iso);
    return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

export default function HistoryDetailScreen() {
    const navigation = useNavigation<NavProp>();
    const route = useRoute<RoutePropType>();
    const { session } = route.params;

    const data = session.full_data;

    // Audio
    const [playingIndex, setPlayingIndex] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState<number | null>(null);
    const soundRef = useRef<Audio.Sound | null>(null);

    // Severity tab
    const [activeTab, setActiveTab] = useState<SeverityTab>('major');

    // Goal linking modal
    const [goalModalOpen, setGoalModalOpen] = useState(false);
    const [goals, setGoals] = useState<Goal[]>([]);
    const [linkedGoalId, setLinkedGoalId] = useState<string | undefined>(session.linked_goal_id);
    const [linkedGoalName, setLinkedGoalName] = useState<string | undefined>();

    // Load goals for the modal
    useFocusEffect(useCallback(() => {
        loadGoals().then(g => {
            setGoals(g);
            if (linkedGoalId) {
                const found = g.find(x => x.id === linkedGoalId);
                setLinkedGoalName(found?.name);
            }
        });
    }, [linkedGoalId]));

    // Cleanup audio on unmount
    useEffect(() => {
        return () => {
            soundRef.current?.unloadAsync();
        };
    }, []);

    const handlePlayAudio = async (audioUrl: string, index: number) => {
        try {
            if (soundRef.current) {
                await soundRef.current.unloadAsync();
                soundRef.current = null;
            }
            if (playingIndex === index) {
                setPlayingIndex(null);
                return;
            }
            setIsLoading(index);
            const sound = new Audio.Sound();
            await sound.loadAsync({ uri: audioUrl });
            soundRef.current = sound;
            sound.setOnPlaybackStatusUpdate((status: any) => {
                if (status.didJustFinish) setPlayingIndex(null);
            });
            await sound.playAsync();
            setPlayingIndex(index);
            setIsLoading(null);
        } catch {
            setIsLoading(null);
        }
    };

    const handleLinkGoal = async (goal: Goal) => {
        try {
            await addRunToGoal(goal.id, {
                date: session.date,
                score: session.score,
                improvement_delta: session.improvement_delta,
            });
            await updateHistorySession(session.id, { linked_goal_id: goal.id });
            setLinkedGoalId(goal.id);
            setLinkedGoalName(goal.name);
            setGoalModalOpen(false);
            Alert.alert('Linked!', `Session added to "${goal.name}".`);
        } catch {
            Alert.alert('Error', 'Failed to link session to goal.');
        }
    };

    const score = session.score;
    const delta = session.improvement_delta;
    const hasDelta = delta !== null;
    const deltaPositive = (delta ?? 0) >= 0;
    const deltaColor = deltaPositive ? Colors.success : Colors.error;

    const baseScore = hasDelta ? Math.max(0, score - Math.abs(delta!)) : score;
    const basePct = (baseScore / 100) * BAR_WIDTH;
    const deltaPct = hasDelta ? (Math.abs(delta!) / 100) * BAR_WIDTH : 0;

    const grade = score >= 90 ? 'Elite' : score >= 75 ? 'Good' : score >= 60 ? 'Progress' : 'Keep Going';
    const gradeColor = GRADE_COLORS[grade] ?? '#00E5A0';

    const feedbackPoints: FeedbackPoint[] = data?.feedback_points ?? [];
    const feedbackByLevel = {
        major: feedbackPoints.filter(fp => fp.severity === 'major'),
        intermediate: feedbackPoints.filter(fp => fp.severity === 'intermediate'),
        minor: feedbackPoints.filter(fp => fp.severity === 'minor'),
    };
    const activeFeedback = feedbackByLevel[activeTab];

    useEffect(() => {
        if (feedbackByLevel.major.length > 0) setActiveTab('major');
        else if (feedbackByLevel.intermediate.length > 0) setActiveTab('intermediate');
        else if (feedbackByLevel.minor.length > 0) setActiveTab('minor');
    }, [data]);

    // Sorted goals: matching activityType first
    const sortedGoals = [...goals].sort((a, b) => {
        const aMatch = a.activityType === session.activityType ? -1 : 0;
        const bMatch = b.activityType === session.activityType ? -1 : 0;
        return aMatch - bMatch;
    });

    return (
        <View style={S.container}>
            <ScrollView
                style={S.screen}
                contentContainerStyle={S.scroll}
                showsVerticalScrollIndicator={false}
            >
                {/* ── Back + Header ── */}
                <View style={S.topBar}>
                    <TouchableOpacity onPress={() => navigation.goBack()} style={S.backBtn} activeOpacity={0.7}>
                        <Text style={S.backArrow}>‹</Text>
                        <Text style={S.backLabel}>History</Text>
                    </TouchableOpacity>
                    <View style={[S.gradePill, { backgroundColor: gradeColor + '22', borderColor: gradeColor + '77' }]}>
                        <Text style={[S.gradeText, { color: gradeColor }]}>{grade}</Text>
                    </View>
                </View>

                <Text style={S.title}>{session.activityType}</Text>
                <Text style={S.subtitle}>{formatDate(session.date)} · {formatTime(session.date)}</Text>

                {session.positive_note ? (
                    <Text style={S.positiveNote}>{session.positive_note}</Text>
                ) : null}

                {/* ── Score Card ── */}
                <View style={S.scoreCard}>
                    <View style={S.scoreRow}>
                        <View>
                            <Text style={S.scoreLabel}>FORM SCORE</Text>
                            <View style={S.scoreNumRow}>
                                <Text style={S.scoreNum}>{score}</Text>
                                <Text style={S.scoreMax}>/100</Text>
                            </View>
                        </View>
                        {hasDelta && (
                            <View style={[S.deltaBadge, { backgroundColor: deltaPositive ? Colors.primaryDim : Colors.errorDim, borderColor: deltaColor }]}>
                                <Text style={[S.deltaBadgeText, { color: deltaColor }]}>
                                    {deltaPositive ? '▲' : '▼'} {delta! > 0 ? '+' : ''}{delta} pts
                                </Text>
                            </View>
                        )}
                    </View>

                    {/* Static bar (no animation needed for history) */}
                    <View style={S.track}>
                        <View style={[S.fillBase, { width: basePct }]} />
                        {hasDelta && (
                            <View style={[S.fillDelta, { width: deltaPct, backgroundColor: deltaColor }]} />
                        )}
                    </View>
                    <View style={S.ticks}>
                        {[0, 25, 50, 75, 100].map(t => (
                            <Text key={t} style={S.tick}>{t}</Text>
                        ))}
                    </View>
                    {hasDelta && (
                        <Text style={[S.deltaCaption, { color: deltaColor }]}>
                            {deltaPositive ? '▲' : '▼'} {Math.abs(delta!)} pts vs previous attempt
                        </Text>
                    )}
                </View>

                {/* ── Link to Goal ── */}
                <TouchableOpacity
                    style={[S.linkGoalBtn, linkedGoalId && S.linkGoalBtnLinked]}
                    onPress={() => setGoalModalOpen(true)}
                    activeOpacity={0.8}
                >
                    <Text style={[S.linkGoalText, linkedGoalId && S.linkGoalTextLinked]}>
                        {linkedGoalId
                            ? `Linked to goal: ${linkedGoalName ?? '…'} · Change`
                            : '+ Link to a Goal'}
                    </Text>
                </TouchableOpacity>

                {/* ── Feedback Section ── */}
                {feedbackPoints.length > 0 && (
                    <View style={S.feedbackSection}>
                        <Text style={S.feedbackTitle}>Coaching Feedback</Text>

                        {/* Severity tabs */}
                        <View style={S.tabContainer}>
                            {(['major', 'intermediate', 'minor'] as const).map(tab => {
                                const count = feedbackByLevel[tab].length;
                                const isActive = activeTab === tab;
                                const tabColor = tab === 'major' ? '#F07070' : tab === 'intermediate' ? '#F5C84A' : '#90EDB0';
                                return (
                                    <TouchableOpacity
                                        key={tab}
                                        style={[S.tabBtn, isActive && { backgroundColor: tabColor + '22', borderColor: tabColor + '99', borderBottomColor: tabColor }]}
                                        onPress={() => setActiveTab(tab)}
                                        activeOpacity={0.7}
                                    >
                                        <Text style={[S.tabLabel, isActive && { color: tabColor }]}>
                                            {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                        </Text>
                                        {count > 0 && (
                                            <View style={[S.tabBadge, { backgroundColor: tabColor + '33' }]}>
                                                <Text style={[S.tabBadgeText, { color: tabColor }]}>{count}</Text>
                                            </View>
                                        )}
                                    </TouchableOpacity>
                                );
                            })}
                        </View>

                        {activeFeedback.length === 0 && (
                            <Text style={S.noFeedback}>No {activeTab} issues flagged.</Text>
                        )}

                        {activeFeedback.map((fp, i) => {
                            // Index in the original full list for audio state tracking
                            const globalIdx = feedbackPoints.indexOf(fp);
                            const isPlaying = playingIndex === globalIdx;
                            const loading = isLoading === globalIdx;
                            return (
                                <View key={globalIdx} style={S.feedbackCard}>
                                    {fp.positive_note ? (
                                        <Text style={S.feedbackPositiveNote}>{fp.positive_note}</Text>
                                    ) : null}
                                    <View style={S.feedbackCardTop}>
                                        <Text style={S.feedbackIndex}>{i + 1}</Text>
                                        <Text style={S.feedbackText} numberOfLines={5}>
                                            {fp.coaching_script}
                                        </Text>
                                    </View>
                                    {fp.audio_url ? (
                                        <TouchableOpacity
                                            style={[S.audioBtn, isPlaying && S.audioBtnActive]}
                                            onPress={() => handlePlayAudio(fp.audio_url, globalIdx)}
                                            activeOpacity={0.8}
                                            disabled={loading}
                                        >
                                            {loading ? (
                                                <ActivityIndicator size="small" color={Colors.primary} />
                                            ) : (
                                                <Text style={[S.audioBtnText, isPlaying && S.audioBtnTextActive]}>
                                                    {isPlaying ? '⏸  Stop' : '▶  Play Audio'}
                                                </Text>
                                            )}
                                        </TouchableOpacity>
                                    ) : null}
                                </View>
                            );
                        })}
                    </View>
                )}

                {feedbackPoints.length === 0 && !data && (
                    <View style={S.noDataBox}>
                        <Text style={S.noDataText}>
                            Detailed feedback not available for this session.{'\n'}
                            Sessions recorded after this update will include full playback.
                        </Text>
                    </View>
                )}
            </ScrollView>

            {/* ── Goal Linking Modal ── */}
            <Modal
                visible={goalModalOpen}
                animationType="slide"
                transparent
                onRequestClose={() => setGoalModalOpen(false)}
            >
                <TouchableOpacity
                    style={S.modalOverlay}
                    activeOpacity={1}
                    onPress={() => setGoalModalOpen(false)}
                />
                <View style={S.modalSheet}>
                    <View style={S.modalHandle} />
                    <Text style={S.modalTitle}>Link to a Goal</Text>
                    <Text style={S.modalSubtitle}>
                        This adds this session's score to the goal's history.
                    </Text>

                    {sortedGoals.length === 0 && (
                        <Text style={S.noGoalsText}>
                            No goals yet. Create one on the Goals tab first.
                        </Text>
                    )}

                    <FlatList
                        data={sortedGoals}
                        keyExtractor={g => g.id}
                        style={{ maxHeight: 360 }}
                        renderItem={({ item: goal }) => {
                            const isMatch = goal.activityType === session.activityType;
                            const isLinked = linkedGoalId === goal.id;
                            return (
                                <TouchableOpacity
                                    style={[S.goalItem, isLinked && S.goalItemLinked]}
                                    onPress={() => handleLinkGoal(goal)}
                                    activeOpacity={0.75}
                                >
                                    <View style={{ flex: 1 }}>
                                        <View style={S.goalItemHeader}>
                                            <Text style={[S.goalItemActivity, isMatch && { color: Colors.primary }]}>
                                                {goal.activityType}
                                            </Text>
                                            {isMatch && (
                                                <View style={S.matchPill}>
                                                    <Text style={S.matchPillText}>Match</Text>
                                                </View>
                                            )}
                                        </View>
                                        <Text style={S.goalItemName} numberOfLines={2}>{goal.name}</Text>
                                        <Text style={S.goalItemMeta}>
                                            {goal.runs.length} session{goal.runs.length !== 1 ? 's' : ''}
                                        </Text>
                                    </View>
                                    {isLinked && <Text style={S.goalItemCheckmark}>✓</Text>}
                                </TouchableOpacity>
                            );
                        }}
                    />
                    <TouchableOpacity style={S.modalCancel} onPress={() => setGoalModalOpen(false)} activeOpacity={0.7}>
                        <Text style={S.modalCancelText}>Cancel</Text>
                    </TouchableOpacity>
                </View>
            </Modal>
        </View>
    );
}

const S = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.background },
    screen: { flex: 1 },
    scroll: {
        paddingHorizontal: Spacing.lg,
        paddingTop: 54,
        paddingBottom: 48,
    },

    // Header
    topBar: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.md,
    },
    backBtn: { flexDirection: 'row', alignItems: 'center', gap: 4 },
    backArrow: { color: Colors.primary, fontSize: 28, lineHeight: 30, fontWeight: '300' },
    backLabel: { color: Colors.primary, fontSize: 16, fontWeight: '600' },
    gradePill: {
        borderRadius: Radius.full,
        borderWidth: 1,
        paddingHorizontal: 12,
        paddingVertical: 4,
    },
    gradeText: { fontSize: 11, fontWeight: '800', letterSpacing: 0.8, textTransform: 'uppercase' },

    title: {
        color: Colors.text,
        fontSize: 26,
        fontWeight: '800',
        letterSpacing: -0.8,
        marginBottom: 4,
    },
    subtitle: {
        color: Colors.textMuted,
        fontSize: 13,
        marginBottom: Spacing.sm,
    },
    positiveNote: {
        color: Colors.primary,
        fontSize: 14,
        fontStyle: 'italic',
        marginBottom: Spacing.lg,
        lineHeight: 20,
    },

    // Score
    scoreCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.lg,
        marginBottom: Spacing.md,
    },
    scoreRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        marginBottom: Spacing.md,
    },
    scoreLabel: {
        color: Colors.textSecondary,
        fontSize: 10,
        fontWeight: '700',
        letterSpacing: 1.5,
        textTransform: 'uppercase',
        marginBottom: 4,
    },
    scoreNumRow: { flexDirection: 'row', alignItems: 'flex-end', gap: 4 },
    scoreNum: { color: Colors.text, fontSize: 52, fontWeight: '800', lineHeight: 56, letterSpacing: -2 },
    scoreMax: { color: Colors.textMuted, fontSize: 18, fontWeight: '600', paddingBottom: 8 },
    deltaBadge: {
        borderRadius: Radius.md,
        borderWidth: 1,
        paddingHorizontal: 12,
        paddingVertical: 7,
        alignSelf: 'flex-start',
    },
    deltaBadgeText: { fontSize: 14, fontWeight: '800' },
    track: {
        height: 8,
        backgroundColor: Colors.backgroundAlt,
        borderRadius: 4,
        flexDirection: 'row',
        overflow: 'hidden',
        marginBottom: 6,
    },
    fillBase: { height: 8, backgroundColor: Colors.primary, borderRadius: 4 },
    fillDelta: { height: 8, borderRadius: 4 },
    ticks: { flexDirection: 'row', justifyContent: 'space-between' },
    tick: { color: Colors.textMuted, fontSize: 10 },
    deltaCaption: { fontSize: 12, fontWeight: '700', marginTop: 8, textAlign: 'center' },

    // Link to goal
    linkGoalBtn: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1.5,
        borderColor: Colors.primaryBorder,
        borderStyle: 'dashed',
        paddingVertical: 14,
        alignItems: 'center',
        marginBottom: Spacing.md,
    },
    linkGoalBtnLinked: {
        borderStyle: 'solid',
        backgroundColor: Colors.primaryDim,
    },
    linkGoalText: {
        color: Colors.primary,
        fontSize: 14,
        fontWeight: '700',
    },
    linkGoalTextLinked: {
        color: Colors.primary,
    },

    // Feedback
    feedbackSection: { marginTop: Spacing.sm },
    feedbackTitle: {
        color: Colors.text,
        fontSize: 18,
        fontWeight: '800',
        letterSpacing: -0.4,
        marginBottom: Spacing.md,
    },
    tabContainer: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: Spacing.md,
    },
    tabBtn: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 5,
        paddingVertical: 9,
        borderRadius: Radius.md,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        backgroundColor: Colors.surface,
        borderBottomWidth: 2,
    },
    tabLabel: {
        color: Colors.textSecondary,
        fontSize: 11,
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: 0.6,
    },
    tabBadge: {
        borderRadius: Radius.full,
        paddingHorizontal: 6,
        paddingVertical: 1,
        minWidth: 18,
        alignItems: 'center',
    },
    tabBadgeText: { fontSize: 10, fontWeight: '800' },
    noFeedback: {
        color: Colors.textMuted,
        fontSize: 13,
        textAlign: 'center',
        paddingVertical: Spacing.xl,
    },
    feedbackCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
    },
    feedbackCardTop: {
        flexDirection: 'row',
        gap: Spacing.sm,
        marginBottom: Spacing.sm,
    },
    feedbackIndex: {
        color: Colors.primary,
        fontSize: 11,
        fontWeight: '800',
        width: 18,
        marginTop: 2,
    },
    feedbackPositiveNote: {
        fontSize: 13,
        color: Colors.success,
        fontWeight: '600',
        marginBottom: 6,
    },
    feedbackText: {
        flex: 1,
        color: Colors.text,
        fontSize: 14,
        lineHeight: 21,
    },
    audioBtn: {
        paddingVertical: 9,
        paddingHorizontal: Spacing.md,
        borderRadius: Radius.md,
        borderWidth: 1,
        borderColor: Colors.primaryBorder,
        backgroundColor: Colors.backgroundAlt,
        alignItems: 'center',
    },
    audioBtnActive: {
        backgroundColor: Colors.primaryDim,
        borderColor: Colors.primary,
    },
    audioBtnText: {
        color: Colors.primary,
        fontSize: 13,
        fontWeight: '700',
    },
    audioBtnTextActive: {
        color: Colors.primary,
    },
    noDataBox: {
        marginTop: Spacing.xl,
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.lg,
    },
    noDataText: {
        color: Colors.textMuted,
        fontSize: 13,
        lineHeight: 20,
        textAlign: 'center',
    },

    // Goal modal
    modalOverlay: {
        position: 'absolute',
        top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.55)',
    },
    modalSheet: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        backgroundColor: Colors.surface,
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        paddingHorizontal: Spacing.lg,
        paddingTop: Spacing.md,
        paddingBottom: 40,
        borderTopWidth: 1,
        borderColor: Colors.glassBorder,
    },
    modalHandle: {
        width: 40,
        height: 4,
        borderRadius: 2,
        backgroundColor: Colors.glassBorder,
        alignSelf: 'center',
        marginBottom: Spacing.md,
    },
    modalTitle: {
        color: Colors.text,
        fontSize: 20,
        fontWeight: '800',
        marginBottom: 4,
    },
    modalSubtitle: {
        color: Colors.textMuted,
        fontSize: 13,
        marginBottom: Spacing.lg,
    },
    noGoalsText: {
        color: Colors.textMuted,
        fontSize: 13,
        textAlign: 'center',
        paddingVertical: Spacing.xl,
    },
    goalItem: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: Colors.backgroundAlt,
        borderRadius: Radius.md,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
    },
    goalItemLinked: {
        borderColor: Colors.primaryBorder,
        backgroundColor: Colors.primaryDim,
    },
    goalItemHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        marginBottom: 3,
    },
    goalItemActivity: {
        fontSize: 10,
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: 1,
        color: Colors.textSecondary,
    },
    matchPill: {
        backgroundColor: Colors.primaryDim,
        borderRadius: Radius.full,
        paddingHorizontal: 7,
        paddingVertical: 2,
    },
    matchPillText: {
        color: Colors.primary,
        fontSize: 9,
        fontWeight: '800',
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    goalItemName: {
        color: Colors.text,
        fontSize: 14,
        fontWeight: '700',
        marginBottom: 2,
    },
    goalItemMeta: {
        color: Colors.textMuted,
        fontSize: 12,
    },
    goalItemCheckmark: {
        color: Colors.primary,
        fontSize: 20,
        fontWeight: '800',
        marginLeft: Spacing.sm,
    },
    modalCancel: {
        marginTop: Spacing.md,
        alignItems: 'center',
        paddingVertical: 14,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
    },
    modalCancelText: {
        color: Colors.textSecondary,
        fontSize: 15,
        fontWeight: '600',
    },
});
