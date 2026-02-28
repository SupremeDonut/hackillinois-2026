import React, { useEffect, useRef, useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, ScrollView, ActivityIndicator } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Audio, Video, ResizeMode } from 'expo-av';
import { RootStackParamList } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { addRunToGoal } from '../services/goalStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Complete'>;
type CompleteRouteProp = RouteProp<RootStackParamList, 'Complete'>;

const BAR_WIDTH = 280;
type SeverityTab = 'major' | 'intermediate' | 'minor';

export default function CompleteScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<CompleteRouteProp>();
    const { data, activityType, goalId } = route.params;
    const [activeTab, setActiveTab] = useState<SeverityTab>('major');
    
    // Audio playback state
    const [playingIndex, setPlayingIndex] = useState<number | null>(null);
    const [isLoading, setIsLoading] = useState<number | null>(null);
    const soundRef = useRef<Audio.Sound | null>(null);

    // Video playback state
    const [expandedVideoIndex, setExpandedVideoIndex] = useState<number | null>(null);
    const videoRefs = useRef<{ [key: number]: any }>({});

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

    useEffect(() => {
        // Save run to goal if this session is associated with one
        if (goalId) {
            addRunToGoal(goalId, {
                date: new Date().toISOString(),
                score: data.progress_score,
                improvement_delta: data.improvement_delta ?? null,
            }).catch((e) => console.warn('[Goals] Failed to save run:', e));
        }

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

    // Handle video playback
    const handlePlayVideo = async (videoUrl: string, timestamp: number, feedbackIndex: number) => {
        try {
            if (expandedVideoIndex === feedbackIndex) {
                setExpandedVideoIndex(null);
                return;
            }

            setExpandedVideoIndex(feedbackIndex);

            // Seek to mistake timestamp after a brief delay for the video to load
            setTimeout(() => {
                const videoRef = videoRefs.current[feedbackIndex];
                if (videoRef) {
                    videoRef.playAsync();
                }
            }, 500);
        } catch (error) {
            console.error('Video playback error:', error);
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
                    <Text style={S.gradeText}>{grade}</Text>
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
                                    <Text
                                        style={[
                                            S.tabButtonText,
                                            activeTab === tab && S.tabButtonTextActive,
                                        ]}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)} ({feedbackByLevel[tab].length})
                                    </Text>
                                </TouchableOpacity>
                            ))}
                        </View>

                        {/* Feedback List */}
                        <View style={S.feedbackList}>
                            {activeFeedback.length > 0 ? (
                                activeFeedback.map((feedback: any, idx: number) => (
                                    <View key={idx} style={S.feedbackCard}>
                                        {/* Video Player - Expanded */}
                                        {expandedVideoIndex === idx && feedback.video_url && (
                                            <View style={S.videoContainer}>
                                                <Video
                                                    ref={(ref) => {
                                                        videoRefs.current[idx] = ref;
                                                    }}
                                                    source={{ uri: feedback.video_url }}
                                                    style={S.videoPlayer}
                                                    resizeMode={ResizeMode.CONTAIN}
                                                    useNativeControls
                                                    onLoadStart={() => {
                                                        if (videoRefs.current[idx]) {
                                                            videoRefs.current[idx].pauseAsync();
                                                            videoRefs.current[idx].setPositionAsync(feedback.mistake_timestamp_ms);
                                                        }
                                                    }}
                                                    onError={(error) => {
                                                        console.error('Video error:', error);
                                                    }}
                                                />
                                                <Text style={S.videoTimestamp}>
                                                    @{(feedback.mistake_timestamp_ms / 1000).toFixed(1)}s
                                                </Text>
                                            </View>
                                        )}

                                        {/* Feedback Card Header */}
                                        <View style={S.feedbackCardHeader}>
                                            <Text style={S.feedbackCoachingScript}>{feedback.coaching_script}</Text>
                                            <View style={S.buttonGroup}>
                                                {feedback.video_url && (
                                                    <TouchableOpacity
                                                        style={[
                                                            S.actionButton,
                                                            expandedVideoIndex === idx && S.actionButtonActive,
                                                        ]}
                                                        onPress={() => handlePlayVideo(feedback.video_url, feedback.mistake_timestamp_ms, idx)}
                                                        activeOpacity={0.7}
                                                    >
                                                        <Text style={S.actionButtonText}>
                                                            {expandedVideoIndex === idx ? '✕' : '▶'}
                                                        </Text>
                                                    </TouchableOpacity>
                                                )}
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
                                                        ) : (
                                                            <Text style={S.actionButtonText}>
                                                                {playingIndex === idx ? '⏸' : '♪'}
                                                            </Text>
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
                    onPress={() => navigation.navigate('Recording', { activityType, previousData: data })}
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
    gradeText: {
        fontSize: 12,
        fontWeight: '600',
        color: Colors.primary,
        marginBottom: Spacing.sm,
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
        fontSize: 14,
        fontWeight: '700',
        color: Colors.textSecondary,
        letterSpacing: 1.2,
        marginBottom: Spacing.md,
    },
    tabContainer: {
        flexDirection: 'row',
        gap: Spacing.sm,
        marginBottom: Spacing.md,
    },
    tabButton: {
        flex: 1,
        paddingVertical: 8,
        paddingHorizontal: 12,
        borderRadius: Radius.md,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        backgroundColor: Colors.surface,
        alignItems: 'center',
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
    },
    videoContainer: {
        backgroundColor: '#000',
        borderRadius: Radius.md,
        overflow: 'hidden',
        marginBottom: Spacing.md,
        height: 200,
    },
    videoPlayer: {
        width: '100%',
        height: '100%',
    },
    videoTimestamp: {
        position: 'absolute',
        bottom: Spacing.sm,
        left: Spacing.sm,
        backgroundColor: 'rgba(0,0,0,0.8)',
        color: Colors.background,
        paddingHorizontal: Spacing.sm,
        paddingVertical: 4,
        borderRadius: Radius.sm,
        fontSize: 11,
        fontWeight: '600',
    },
    feedbackCardHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        gap: Spacing.sm,
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
