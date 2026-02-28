import React, { useState, useCallback } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, TextInput,
    KeyboardAvoidingView, Platform, ScrollView, Alert,
} from 'react-native';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, Goal } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { loadGoals, createGoal } from '../services/goalStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

const ACTIVITIES = [
    'Basketball Shot',
    'Golf Swing',
    'Badminton Smash',
    'Tennis Serve',
    'Guitar Chord',
    'Dance Move',
];

export default function HomeScreen() {
    const navigation = useNavigation<NavigationProp>();
    const [activityType, setActivityType] = useState('');
    const [description, setDescription] = useState('');
    const [goals, setGoals] = useState<Goal[]>([]);
    const [newGoalText, setNewGoalText] = useState('');
    const [addingGoal, setAddingGoal] = useState(false);
    const isReady = activityType.trim().length > 0;

    // Reload goals whenever screen is focused (e.g. coming back from GoalDetail)
    useFocusEffect(
        useCallback(() => {
            loadGoals().then(setGoals);
        }, []),
    );

    const handleAddGoal = async () => {
        if (!newGoalText.trim()) return;
        if (!activityType.trim()) {
            Alert.alert('Select activity', 'Please select an activity first so the goal can be tracked correctly.');
            return;
        }
        await createGoal(newGoalText.trim(), activityType);
        setNewGoalText('');
        setAddingGoal(false);
        loadGoals().then(setGoals);
    };

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={{ flex: 1, backgroundColor: Colors.background }}
        >
            <ScrollView
                contentContainerStyle={S.scroll}
                keyboardShouldPersistTaps="handled"
                showsVerticalScrollIndicator={false}
            >
                {/* ── Brand ── */}
                <View style={S.brand}>
                    <View style={S.logoMark}>
                        <View style={S.logoInner} />
                    </View>
                    <Text style={S.appName}>MotionCoach</Text>
                    <Text style={S.tagline}>Coaching for every hobby.</Text>
                </View>

                {/* ── Input Card ── */}
                <View style={S.card}>
                    <Text style={S.inputLabel}>What are you practicing?</Text>
                    <TextInput
                        style={S.input}
                        value={activityType}
                        onChangeText={setActivityType}
                        placeholder="e.g. basketball free throw"
                        placeholderTextColor={Colors.textMuted}
                        returnKeyType="done"
                    />
                    <View style={S.chips}>
                        {ACTIVITIES.map((label) => {
                            const active = activityType === label;
                            return (
                                <TouchableOpacity
                                    key={label}
                                    style={[S.chip, active && S.chipActive]}
                                    onPress={() => setActivityType(label)}
                                    activeOpacity={0.7}
                                >
                                    <Text style={[S.chipLabel, active && S.chipLabelActive]}>
                                        {label}
                                    </Text>
                                </TouchableOpacity>
                            );
                        })}
                    </View>
                </View>

                {isReady && (
                    <View style={S.card}>
                        <Text style={S.inputLabel}>Any specific focus? (Optional)</Text>
                        <TextInput
                            style={[S.input, S.textArea]}
                            value={description}
                            onChangeText={setDescription}
                            placeholder={`e.g. I want to improve my ${activityType.toLowerCase() || 'form'}.`}
                            placeholderTextColor={Colors.textMuted}
                            multiline
                            textAlignVertical="top"
                        />
                    </View>
                )}

                {/* ── CTA ── */}
                <TouchableOpacity
                    style={[S.cta, !isReady && S.ctaDisabled]}
                    onPress={() => isReady && navigation.navigate('Recording', {
                        activityType,
                        description: description.trim() || `I want to improve my ${activityType} form.`
                    })}
                    disabled={!isReady}
                    activeOpacity={0.85}
                >
                    <Text style={[S.ctaText, !isReady && S.ctaTextDisabled]}>
                        Start Recording
                    </Text>
                </TouchableOpacity>

                <Text style={S.hint}>Max 5 seconds of footage</Text>

                {/* ── Goals Section ── */}
                <View style={S.goalsHeader}>
                    <Text style={S.goalsTitle}>My Goals</Text>
                    <TouchableOpacity
                        style={S.addGoalBtn}
                        onPress={() => setAddingGoal((v) => !v)}
                        activeOpacity={0.7}
                    >
                        <Text style={S.addGoalBtnText}>{addingGoal ? 'Cancel' : '+ New Goal'}</Text>
                    </TouchableOpacity>
                </View>

                {addingGoal && (
                    <View style={S.card}>
                        <Text style={S.inputLabel}>Goal description</Text>
                        <TextInput
                            style={S.input}
                            value={newGoalText}
                            onChangeText={setNewGoalText}
                            placeholder={activityType ? `e.g. Nail my ${activityType} follow-through` : 'Select an activity first'}
                            placeholderTextColor={Colors.textMuted}
                            returnKeyType="done"
                            onSubmitEditing={handleAddGoal}
                            autoFocus
                        />
                        <TouchableOpacity
                            style={[S.cta, !newGoalText.trim() && S.ctaDisabled]}
                            onPress={handleAddGoal}
                            disabled={!newGoalText.trim()}
                            activeOpacity={0.85}
                        >
                            <Text style={[S.ctaText, !newGoalText.trim() && S.ctaTextDisabled]}>
                                Add Goal
                            </Text>
                        </TouchableOpacity>
                    </View>
                )}

                {goals.length === 0 && !addingGoal && (
                    <Text style={S.goalsEmpty}>
                        No goals yet. Add one to track your progress over time.
                    </Text>
                )}

                {goals.map((goal) => {
                    const latestRun = goal.runs[goal.runs.length - 1];
                    const hasDelta = latestRun?.improvement_delta !== null && latestRun?.improvement_delta !== undefined;
                    const positive = (latestRun?.improvement_delta ?? 0) > 0;
                    return (
                        <TouchableOpacity
                            key={goal.id}
                            style={S.goalCard}
                            onPress={() => navigation.navigate('GoalDetail', { goalId: goal.id })}
                            activeOpacity={0.75}
                        >
                            <View style={S.goalCardLeft}>
                                <Text style={S.goalCardActivity}>{goal.activityType}</Text>
                                <Text style={S.goalCardName} numberOfLines={2}>{goal.name}</Text>
                                <Text style={S.goalCardMeta}>
                                    {goal.runs.length} session{goal.runs.length !== 1 ? 's' : ''}
                                    {latestRun ? ` · Last score: ${latestRun.score}` : ''}
                                </Text>
                            </View>
                            <View style={S.goalCardRight}>
                                {latestRun ? (
                                    <Text style={S.goalCardScore}>{latestRun.score}</Text>
                                ) : (
                                    <Text style={S.goalCardNoScore}>—</Text>
                                )}
                                {hasDelta && (
                                    <Text style={[S.goalCardDelta, { color: positive ? Colors.success : Colors.error }]}>
                                        {positive ? '▲' : '▼'} {Math.abs(latestRun!.improvement_delta!)}
                                    </Text>
                                )}
                                <Text style={S.goalCardArrow}>›</Text>
                            </View>
                        </TouchableOpacity>
                    );
                })}
            </ScrollView>
        </KeyboardAvoidingView>
    );
}

const S = StyleSheet.create({
    scroll: {
        flexGrow: 1,
        paddingHorizontal: Spacing.lg,
        paddingTop: 80,
        paddingBottom: 48,
        backgroundColor: Colors.background,
    },
    brand: {
        alignItems: 'center',
        marginBottom: 48,
    },
    logoMark: {
        width: 52,
        height: 52,
        borderRadius: 16,
        backgroundColor: Colors.primary,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 14,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.35,
        shadowRadius: 16,
        elevation: 10,
    },
    logoInner: {
        width: 22,
        height: 22,
        borderRadius: 4,
        backgroundColor: Colors.background,
        opacity: 0.85,
    },
    appName: {
        fontSize: 30,
        fontWeight: '800',
        color: Colors.text,
        letterSpacing: -0.8,
        marginBottom: 4,
    },
    tagline: {
        fontSize: 14,
        color: Colors.textSecondary,
    },
    card: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        padding: Spacing.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        marginBottom: Spacing.lg,
    },
    inputLabel: {
        fontSize: 11,
        fontWeight: '700',
        color: Colors.textSecondary,
        textTransform: 'uppercase',
        letterSpacing: 1.2,
        marginBottom: Spacing.sm,
    },
    input: {
        backgroundColor: Colors.backgroundAlt,
        color: Colors.text,
        fontSize: 15,
        paddingHorizontal: Spacing.md,
        paddingVertical: 13,
        borderRadius: Radius.md,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        marginBottom: Spacing.md,
    },
    textArea: {
        height: 80,
    },
    chips: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
    },
    chip: {
        paddingHorizontal: 13,
        paddingVertical: 7,
        borderRadius: Radius.full,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        backgroundColor: Colors.backgroundAlt,
    },
    chipActive: {
        backgroundColor: Colors.primaryDim,
        borderColor: Colors.primaryBorder,
    },
    chipLabel: {
        color: Colors.textSecondary,
        fontSize: 13,
        fontWeight: '500',
    },
    chipLabelActive: {
        color: Colors.primary,
        fontWeight: '700',
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
    ctaDisabled: {
        backgroundColor: Colors.surface,
        shadowOpacity: 0,
        elevation: 0,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
    },
    ctaText: {
        color: Colors.background,
        fontSize: 16,
        fontWeight: '700',
        letterSpacing: 0.2,
    },
    ctaTextDisabled: {
        color: Colors.textMuted,
    },
    hint: {
        color: Colors.textMuted,
        fontSize: 12,
        textAlign: 'center',
        marginTop: Spacing.md,
        marginBottom: Spacing.xl,
    },
    // ── Goals section ──────────────────────────────────────────────────────
    goalsHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: Spacing.md,
    },
    goalsTitle: {
        color: Colors.text,
        fontSize: 18,
        fontWeight: '800',
        letterSpacing: -0.4,
    },
    addGoalBtn: {
        paddingHorizontal: 14,
        paddingVertical: 7,
        borderRadius: Radius.full,
        borderWidth: 1,
        borderColor: Colors.primaryBorder,
        backgroundColor: Colors.primaryDim,
    },
    addGoalBtnText: {
        color: Colors.primary,
        fontSize: 13,
        fontWeight: '700',
    },
    goalsEmpty: {
        color: Colors.textMuted,
        fontSize: 13,
        textAlign: 'center',
        marginTop: Spacing.sm,
        marginBottom: Spacing.xl,
    },
    goalCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
        flexDirection: 'row',
        alignItems: 'center',
    },
    goalCardLeft: { flex: 1, marginRight: Spacing.sm },
    goalCardActivity: {
        color: Colors.primary,
        fontSize: 10,
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: 1.2,
        marginBottom: 3,
    },
    goalCardName: {
        color: Colors.text,
        fontSize: 15,
        fontWeight: '700',
        marginBottom: 4,
    },
    goalCardMeta: {
        color: Colors.textMuted,
        fontSize: 12,
    },
    goalCardRight: { alignItems: 'flex-end' },
    goalCardScore: {
        color: Colors.text,
        fontSize: 22,
        fontWeight: '800',
    },
    goalCardNoScore: {
        color: Colors.textMuted,
        fontSize: 22,
        fontWeight: '800',
    },
    goalCardDelta: {
        fontSize: 12,
        fontWeight: '700',
    },
    goalCardArrow: {
        color: Colors.textSecondary,
        fontSize: 20,
        marginTop: 2,
    },
});
