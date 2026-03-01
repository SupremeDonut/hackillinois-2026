import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, ScrollView, ActivityIndicator } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Audio } from 'expo-av';
import { RootStackParamList } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { addRunToGoal } from '../services/goalStore';
import { addToHistory } from '../services/historyStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Complete'>;
type CompleteRouteProp = RouteProp<RootStackParamList, 'Complete'>;

const BAR_WIDTH = 280;
type SeverityTab = 'major' | 'intermediate' | 'minor';

const GRADE_COLORS: Record<string, string> = {
    Elite: '#00E5A0',
    Good: '#90EDB0',
    Progress: '#F5C84A',
    'Keep Going': '#F07070',
};

const SEVERITY_COLORS: Record<string, string> = {
    major: '#F07070',
    intermediate: '#F5C84A',
    minor: '#90EDB0',
};

export default function CompleteScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<CompleteRouteProp>();
    const { data, activityType, goalId, voiceId } = route.params;
    const [activeTab, setActiveTab] = useState<SeverityTab>('major');
    
    // Audio playback state
    const [playingIndex, setPlayingIndex] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState<number | null>(null);
    const soundRef = useRef<Audio.Sound | null>(null);

    const score = data.progress_score;
    const delta = data.improvement_delta ?? null;
    const hasDelta = delta !== null;
    const deltaPositive = (delta ?? 0) >= 0;

    const baseScore = hasDelta ? Math.max(0, score - Math.abs(delta!)) : score;
    const baseFill = (baseScore / 100) * BAR_WIDTH;
    const deltaFill = hasDelta ? (Math.abs(delta!) / 100) * BAR_WIDTH : 0;

    const baseAnim = useRef(new Animated.Value(0)).current;
    const deltaAnim = useRef(new Animated.Value(0)).current;
    const fadeIn = useRef(new Animated.Value(0)).current;

    // Group feedback points by severity
    const feedbackByLevel = {
        major: (data.feedback_points || []).filter((fp: any) => fp.severity === 'major'),
        intermediate: (data.feedback_points || []).filter((fp: any) => fp.severity === 'intermediate'),
        minor: (data.feedback_points || []).filter((fp: any) => fp.severity === 'minor'),
    };

    const activeFeedback = feedbackByLevel[activeTab];

    // Auto-select the first non-empty severity tab
    useEffect(() => {
        if (feedbackByLevel.major.length > 0) setActiveTab('major');
        else if (feedbackByLevel.intermediate.length > 0) setActiveTab('intermediate');
        else if (feedbackByLevel.minor.length > 0) setActiveTab('minor');
    }, [data]);

    useEffect(() => {
        // Save run to goal if this session is associated with one
        if (goalId) {
            addRunToGoal(goalId, {
                date: new Date().toISOString(),
                score: data.progress_score,
                improvement_delta: data.improvement_delta ?? null,
            }).catch((e) => console.warn('[Goals] Failed to save run:', e));
        }

        // Always save to history
        addToHistory({
            date: new Date().toISOString(),
            activityType,
            score: data.progress_score,
            improvement_delta: data.improvement_delta ?? null,
            positive_note: data.positive_note ?? '',
            feedback_count: (data.feedback_points || []).length,
            full_data: data,
        }).catch((e) => console.warn('[History] Failed to save session:', e));

        Animated.parallel([
            Animated.timing(fadeIn, { toValue: 1, duration: 500, useNativeDriver: true }),
            Animated.sequence([
                Animated.timing(baseAnim, { toValue: baseFill, duration: 900, useNativeDriver: false }),
                Animated.timing(deltaAnim, { toValue: deltaFill, duration: 500, useNativeDriver: false }),
            ]),
        ]).start();

        return () => {
            // Cleanup audio on unmount
            if (soundRef.current) {
                soundRef.current.unloadAsync();
            }
        };
    }, []);

    const deltaColor = deltaPositive ? Colors.success : Colors.error;

    // Score grade label — no emojis
    const grade = score >= 90 ? 'Elite' : score >= 75 ? 'Good' : score >= 60 ? 'Progress' : 'Keep Going';

    // Handle audio playback
    const handlePlayAudio = async (audioUrl: string, feedbackIndex: number) => {
        try {
            // Stop current playback if any
            if (soundRef.current) {
                await soundRef.current.unloadAsync();
                soundRef.current = null;
            }

            // If clicking same item, just toggle off
            if (playingIndex === feedbackIndex) {
                setPlayingIndex(null);
                return;
            }

            // Start loading
            setIsLoading(feedbackIndex);

            // Create and load sound from base64 or URI
            const sound = new Audio.Sound();
            await sound.loadAsync({ uri: audioUrl });
            soundRef.current = sound;

            // Setup playback end listener
            sound.setOnPlaybackStatusUpdate((status: any) => {
                if (status.didJustFinish) {
                    setPlayingIndex(null);
                }
            });

            // Play sound
            await sound.playAsync();
            setPlayingIndex(feedbackIndex);
            setIsLoading(null);
        } catch (error) {
            console.error('Audio playback error:', error);
            setIsLoading(null);
        }
    };

    return (
        <ScrollView
            style={S.screen}
            contentContainerStyle={S.scroll}
            showsVerticalScrollIndicator={false}
        >
            <Animated.View style={[S.content, { opacity: fadeIn }]}>

                {/* ── Header ── */}
                <View style={S.header}>
                    <View style={[S.gradePill, { backgroundColor: (GRADE_COLORS[grade] ?? '#00E5A0') + '22', borderColor: (GRADE_COLORS[grade] ?? '#00E5A0') + '77' }]}>
                        <Text style={[S.gradeText, { color: GRADE_COLORS[grade] ?? '#00E5A0' }]}>{grade}</Text>
                    </View>
                    <Text style={S.title}>Session Complete</Text>
                    {data.positive_note ? (
                        <Text style={S.positiveNote}>{data.positive_note}</Text>
                    ) : null}
                </View>

                {/* ── Score Card ── */}
                <View style={S.scoreCard}>
                    {/* Score number */}
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

                    {/* Animated bar */}
                    <View style={S.track}>
                        <Animated.View style={[S.fillBase, { width: baseAnim }]} />
                        {hasDelta && (
                            <Animated.View style={[S.fillDelta, { width: deltaAnim, backgroundColor: deltaColor }]} />
                        )}
                    </View>

                    {/* Tick marks */}
                    <View style={S.ticks}>
                        {[0, 25, 50, 75, 100].map(t => (
                            <Text key={t} style={S.tick}>{t}</Text>
                        ))}
                    </View>

                    {hasDelta && (
                        <Text style={[S.deltaCaption, { color: deltaColor }]}>
                            {deltaPositive ? '▲' : '▼'} {Math.abs(delta!)} pts vs last attempt
                        </Text>
                    )}
                </View>

                {/* ── Feedback Section ── */}
                {(data.feedback_points || []).length > 0 && (
                    <View style={S.feedbackSection}>
                        <Text style={S.feedbackSectionTitle}>Areas to Improve</Text>

                        {/* Tab Buttons */}
                        <View style={S.tabContainer}>
                            {(['major', 'intermediate', 'minor'] as const).map((tab) => (
                                <TouchableOpacity
                                    key={tab}
                                    style={[
                                        S.tabButton,
                                        activeTab === tab && S.tabButtonActive,
                                    ]}
                                    onPress={() => setActiveTab(tab)}
                                    activeOpacity={0.7}
                                >
                                    <View style={S.tabButtonInner}>
                                        <View style={[S.tabDot, { backgroundColor: activeTab === tab ? '#16161F' : SEVERITY_COLORS[tab] }]} />
                                        <Text
                                            style={[
                                                S.tabButtonText,
                                                activeTab === tab && S.tabButtonTextActive,
                                            ]}
                                        >
                                            {tab.charAt(0).toUpperCase() + tab.slice(1)} ({feedbackByLevel[tab].length})
                                        </Text>
                                    </View>
                                </TouchableOpacity>
                            ))}
                        </View>

                        {/* Feedback List */}
                        <View style={S.feedbackList}>
                            {activeFeedback.length > 0 ? (
                                activeFeedback.map((feedback: any, idx: number) => (
                                    <View key={idx} style={[S.feedbackCard, { borderLeftColor: SEVERITY_COLORS[activeTab], borderLeftWidth: 3 }]}>
                                        {feedback.positive_note ? (
                                            <Text style={S.feedbackPositiveNote}>{feedback.positive_note}</Text>
                                        ) : null}
                                        <View style={S.feedbackCardHeader}>
                                            <Text style={S.feedbackCoachingScript}>{feedback.coaching_script}</Text>
                                            <View style={S.buttonGroup}>
                                                {feedback.audio_url && (
                                                    <TouchableOpacity
                                                        style={[
                                                            S.actionButton,
                                                            playingIndex === idx && S.actionButtonActive,
                                                        ]}
                                                        onPress={() => handlePlayAudio(feedback.audio_url, idx)}
                                                        disabled={isLoading === idx}
                                                        activeOpacity={0.7}
                                                    >
                                                        {isLoading === idx ? (
                                                            <ActivityIndicator size="small" color={Colors.background} />
                                                        ) : playingIndex === idx ? (
                                                            <View style={S.pauseIcon}>
                                                                <View style={S.pauseBar} />
                                                                <View style={S.pauseBar} />
                                                            </View>
                                                        ) : (
                                                            <Text style={S.actionButtonText}>&#9654;</Text>
                                                        )}
                                                    </TouchableOpacity>
                                                )}
                                            </View>
                                        </View>
                                    </View>
                                ))
                            ) : (
                                <Text style={S.noFeedbackText}>No {activeTab} issues found.</Text>
                            )}
                        </View>
                    </View>
                )}

                {/* ── Actions ── */}
                <TouchableOpacity
                    style={S.primaryBtn}
                    onPress={() => navigation.navigate('Recording', { activityType, previousData: data, voiceId: voiceId ?? 's3TPKV1kjDlVtZbl4Ksh' })}
                    activeOpacity={0.85}
                >
                    <Text style={S.primaryBtnText}>Try Again  →</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={S.secondaryBtn}
                    onPress={() => navigation.navigate('Home')}
                    activeOpacity={0.7}
                >
                    <Text style={S.secondaryBtnText}>New Session</Text>
                </TouchableOpacity>

            </Animated.View>
        </ScrollView>
    );
}

const S = StyleSheet.create({
    screen: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    scroll: {
        flexGrow: 1,
        paddingHorizontal: Spacing.lg,
        paddingTop: 60,
        paddingBottom: 48,
        justifyContent: 'center',
    },
    content: {
        alignItems: 'center',
    },
    header: {
        alignItems: 'center',
        marginBottom: Spacing.xl,
    },
    gradePill: {
        borderRadius: 999,
        borderWidth: 1,
        paddingHorizontal: 16,
        paddingVertical: 6,
        marginBottom: 12,
    },
    gradeText: {
        fontSize: 13,
        fontWeight: '800',
        letterSpacing: 1,
        textTransform: 'uppercase',
    },
    title: {
        fontSize: 30,
        fontWeight: '800',
        color: Colors.text,
        letterSpacing: -0.8,
        marginBottom: Spacing.sm,
    },
    positiveNote: {
        fontSize: 15,
        color: Colors.textSecondary,
        textAlign: 'center',
        lineHeight: 22,
        maxWidth: 300,
    },
    scoreCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        padding: Spacing.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        width: '100%',
        marginBottom: Spacing.xl,
    },
    scoreRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-end',
        marginBottom: Spacing.md,
    },
    scoreLabel: {
        fontSize: 10,
        fontWeight: '700',
        color: Colors.textSecondary,
        letterSpacing: 1.5,
        marginBottom: 4,
    },
    scoreNumRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
    },
    scoreNum: {
        fontSize: 48,
        fontWeight: '800',
        color: Colors.text,
        lineHeight: 52,
        letterSpacing: -2,
    },
    scoreMax: {
        fontSize: 18,
        color: Colors.textSecondary,
        marginBottom: 6,
        marginLeft: 3,
    },
    deltaBadge: {
        borderRadius: Radius.full,
        borderWidth: 1,
        paddingHorizontal: 12,
        paddingVertical: 6,
        marginBottom: 6,
    },
    deltaBadgeText: {
        fontSize: 13,
        fontWeight: '800',
    },
    track: {
        height: 10,
        borderRadius: 5,
        backgroundColor: Colors.backgroundAlt,
        flexDirection: 'row',
        overflow: 'hidden',
        marginBottom: 6,
    },
    fillBase: {
        height: '100%',
        backgroundColor: Colors.primary,
    },
    fillDelta: {
        height: '100%',
    },
    ticks: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    tick: {
        fontSize: 10,
        color: Colors.textMuted,
    },
    deltaCaption: {
        fontSize: 12,
        fontWeight: '700',
        marginTop: Spacing.sm,
        textAlign: 'right',
    },
    primaryBtn: {
        width: '100%',
        backgroundColor: Colors.primary,
        paddingVertical: 18,
        borderRadius: Radius.lg,
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.4,
        shadowRadius: 16,
        elevation: 10,
        marginBottom: Spacing.sm,
    },
    primaryBtnText: {
        color: Colors.background,
        fontSize: 17,
        fontWeight: '800',
        letterSpacing: 0.2,
    },
    secondaryBtn: {
        width: '100%',
        paddingVertical: 16,
        borderRadius: Radius.lg,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        backgroundColor: Colors.surface,
    },
    secondaryBtnText: {
        color: Colors.textSecondary,
        fontSize: 16,
        fontWeight: '600',
    },
    feedbackSection: {
        width: '100%',
        marginBottom: Spacing.xl,
    },
    feedbackSectionTitle: {
        fontSize: 17,
        fontWeight: '800',
        color: Colors.text,
        letterSpacing: -0.3,
        marginBottom: Spacing.md,
    },
    tabContainer: {
        flexDirection: 'row',
        gap: Spacing.sm,
        marginBottom: Spacing.md,
    },
    tabButton: {
        flex: 1,
        paddingVertical: 10,
        paddingHorizontal: 12,
        borderRadius: Radius.md,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        backgroundColor: Colors.surface,
        alignItems: 'center',
    },
    tabButtonInner: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 5,
    },
    tabDot: {
        width: 7,
        height: 7,
        borderRadius: 4,
    },
    tabButtonActive: {
        backgroundColor: Colors.primary,
        borderColor: Colors.primary,
    },
    tabButtonText: {
        fontSize: 12,
        fontWeight: '600',
        color: Colors.textSecondary,
    },
    tabButtonTextActive: {
        color: Colors.background,
    },
    feedbackList: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        minHeight: 100,
    },
    feedbackCard: {
        backgroundColor: Colors.backgroundAlt,
        borderRadius: Radius.md,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
        overflow: 'hidden',
    },
    feedbackCardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: Spacing.sm,
    },
    feedbackPositiveNote: {
        fontSize: 13,
        color: Colors.success,
        fontWeight: '600',
        marginBottom: 6,
    },
    feedbackCoachingScript: {
        fontSize: 14,
        color: Colors.text,
        lineHeight: 20,
        flex: 1,
    },
    buttonGroup: {
        flexDirection: 'row',
        gap: Spacing.xs,
    },
    actionButton: {
        width: 40,
        height: 40,
        borderRadius: Radius.full,
        backgroundColor: Colors.primary,
        justifyContent: 'center',
        alignItems: 'center',
        flexShrink: 0,
    },
    actionButtonActive: {
        backgroundColor: Colors.success,
    },
    actionButtonText: {
        fontSize: 16,
        color: Colors.background,
    },
    pauseIcon: {
        flexDirection: 'row',
        gap: 3,
        alignItems: 'center',
        justifyContent: 'center',
    },
    pauseBar: {
        width: 3,
        height: 14,
        borderRadius: 2,
        backgroundColor: Colors.background,
    },
    feedbackAudioLabel: {
        fontSize: 12,
        color: Colors.primary,
        fontWeight: '600',
    },
    noFeedbackText: {
        fontSize: 14,
        color: Colors.textSecondary,
        textAlign: 'center',
        paddingVertical: Spacing.md,
    },
});
