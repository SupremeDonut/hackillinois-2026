import React, { useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated, ScrollView } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Complete'>;
type CompleteRouteProp = RouteProp<RootStackParamList, 'Complete'>;

const BAR_WIDTH = 280;

export default function CompleteScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<CompleteRouteProp>();
    const { data, activityType } = route.params;

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

    useEffect(() => {
        Animated.parallel([
            Animated.timing(fadeIn, { toValue: 1, duration: 500, useNativeDriver: true }),
            Animated.sequence([
                Animated.timing(baseAnim, { toValue: baseFill, duration: 900, useNativeDriver: false }),
                Animated.timing(deltaAnim, { toValue: deltaFill, duration: 500, useNativeDriver: false }),
            ]),
        ]).start();
    }, []);

    const deltaColor = deltaPositive ? Colors.success : Colors.error;

    // Score grade label — no emojis
    const grade = score >= 90 ? 'Elite' : score >= 75 ? 'Good' : score >= 60 ? 'Progress' : 'Keep Going';

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
});
