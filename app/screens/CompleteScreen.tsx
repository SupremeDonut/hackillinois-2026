import React, { useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Animated } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles, Colors } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Complete'>;
type CompleteRouteProp = RouteProp<RootStackParamList, 'Complete'>;

const BAR_WIDTH = 280; // logical width of the full track

export default function CompleteScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<CompleteRouteProp>();
    const { data, activityType } = route.params;

    const score = data.progress_score;            // e.g. 93
    const delta = data.improvement_delta ?? null; // e.g. +8 or null
    const hasDelta = delta !== null;
    const deltaPositive = (delta ?? 0) >= 0;

    // The "base" portion of the bar (previous score = score - delta on retry, or full score on first run)
    const baseScore = hasDelta ? Math.max(0, score - Math.abs(delta!)) : score;
    const baseFill = (baseScore / 100) * BAR_WIDTH;
    const deltaFill = hasDelta ? (Math.abs(delta!) / 100) * BAR_WIDTH : 0;

    // Animated widths
    const baseAnim = useRef(new Animated.Value(0)).current;
    const deltaAnim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        // Animate base bar first, then the delta segment
        Animated.sequence([
            Animated.timing(baseAnim, {
                toValue: baseFill,
                duration: 900,
                useNativeDriver: false,
            }),
            Animated.timing(deltaAnim, {
                toValue: deltaFill,
                duration: 500,
                useNativeDriver: false,
            }),
        ]).start();
    }, []);

    const deltaColor = deltaPositive ? '#7EE8A2' : '#E86060';  // soft rose instead of harsh system red
    const deltaLabel = hasDelta
        ? `${deltaPositive ? '+' : ''}${delta} pts`
        : null;

    return (
        <View style={globalStyles.fullScreen}>
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>Session Complete</Text>
                <Text style={globalStyles.subHeading}>{data.positive_note}</Text>

                {/* ── Progress Bar ── */}
                <View style={S.barSection}>
                    <View style={S.scoreRow}>
                        <Text style={S.scoreLabel}>Form Score</Text>
                        <Text style={S.scoreValue}>{score}<Text style={S.scoreMax}>/100</Text></Text>
                    </View>

                    {/* Track */}
                    <View style={[S.track, { width: BAR_WIDTH }]}>
                        {/* Base fill (green) */}
                        <Animated.View style={[S.fillBase, { width: baseAnim }]} />
                        {/* Delta segment (brighter green or red) */}
                        {hasDelta && (
                            <Animated.View
                                style={[
                                    S.fillDelta,
                                    { width: deltaAnim, backgroundColor: deltaColor },
                                ]}
                            />
                        )}
                    </View>

                    {/* Delta callout label */}
                    {hasDelta && (
                        <View style={S.deltaLabelRow}>
                            <Text style={[S.deltaCallout, { color: deltaColor }]}>
                                {deltaPositive ? '▲' : '▼'} {deltaLabel} vs last attempt
                            </Text>
                        </View>
                    )}

                    {/* Score tick labels */}
                    <View style={[S.tickRow, { width: BAR_WIDTH }]}>
                        {[0, 25, 50, 75, 100].map(t => (
                            <Text key={t} style={S.tick}>{t}</Text>
                        ))}
                    </View>
                </View>

                <TouchableOpacity
                    style={globalStyles.primaryButton}
                    onPress={() => navigation.navigate('Recording', { activityType, previousData: data })}
                >
                    <Text style={globalStyles.buttonText}>Try Again</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={[globalStyles.primaryButton, { backgroundColor: '#333', marginTop: 10 }]}
                    onPress={() => navigation.navigate('Home')}
                >
                    <Text style={globalStyles.buttonText}>New Session</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}

const S = StyleSheet.create({
    barSection: {
        marginVertical: 32,
        alignItems: 'center',
    },
    scoreRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        width: BAR_WIDTH,
        marginBottom: 10,
    },
    scoreLabel: {
        color: Colors.textSecondary,
        fontSize: 13,
        textTransform: 'uppercase',
        fontWeight: 'bold',
    },
    scoreValue: {
        color: Colors.text,
        fontSize: 18,
        fontWeight: 'bold',
    },
    scoreMax: {
        color: Colors.textSecondary,
        fontSize: 13,
        fontWeight: 'normal',
    },
    track: {
        height: 18,
        borderRadius: 9,
        backgroundColor: '#222',
        flexDirection: 'row',
        overflow: 'hidden',
    },
    fillBase: {
        height: '100%',
        backgroundColor: Colors.primary,
        // No borderRadius — track's overflow:hidden clips the outer corners cleanly
    },
    fillDelta: {
        height: '100%',
        // No borderRadius — same reason
    },
    deltaLabelRow: {
        width: BAR_WIDTH,
        alignItems: 'flex-end',
        marginTop: 6,
    },
    deltaCallout: {
        fontSize: 13,
        fontWeight: '700',
    },
    tickRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginTop: 4,
    },
    tick: {
        color: Colors.textSecondary,
        fontSize: 10,
    },
});

