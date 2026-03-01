import React, { useState, useCallback } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, TextInput, ScrollView, Alert,
} from 'react-native';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, Goal } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { loadGoals, createGoal } from '../services/goalStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

const ACTIVITIES = [
    'Basketball Shot', 'Golf Swing', 'Badminton Smash',
    'Tennis Serve', 'Guitar Chord', 'Dance Move',
];

export default function GoalsTab() {
    const navigation = useNavigation<NavigationProp>();
    const [goals, setGoals] = useState<Goal[]>([]);
    const [addingGoal, setAddingGoal] = useState(false);
    const [newGoalText, setNewGoalText] = useState('');
    const [newGoalActivity, setNewGoalActivity] = useState('');

    useFocusEffect(
        useCallback(() => {
            loadGoals().then(setGoals);
        }, []),
    );

    const handleAddGoal = async () => {
        if (!newGoalText.trim()) return;
        if (!newGoalActivity.trim()) {
            Alert.alert('Select activity', 'Please select an activity for this goal.');
            return;
        }
        await createGoal(newGoalText.trim(), newGoalActivity);
        setNewGoalText('');
        setNewGoalActivity('');
        setAddingGoal(false);
        loadGoals().then(setGoals);
    };

    return (
        <ScrollView
            style={S.screen}
            contentContainerStyle={S.scroll}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
        >
            {/* Header */}
            <View style={S.header}>
                <Text style={S.title}>My Goals</Text>
                <TouchableOpacity
                    style={S.addBtn}
                    onPress={() => setAddingGoal(v => !v)}
                    activeOpacity={0.7}
                >
                    <Text style={S.addBtnText}>{addingGoal ? 'Cancel' : '+ New Goal'}</Text>
                </TouchableOpacity>
            </View>

            {/* Add Goal Form */}
            {addingGoal && (
                <View style={S.card}>
                    <Text style={S.label}>Activity</Text>
                    <View style={S.chips}>
                        {ACTIVITIES.map(a => {
                            const active = newGoalActivity === a;
                            return (
                                <TouchableOpacity
                                    key={a}
                                    style={[S.chip, active && S.chipActive]}
                                    onPress={() => setNewGoalActivity(a)}
                                    activeOpacity={0.7}
                                >
                                    <Text style={[S.chipLabel, active && S.chipLabelActive]}>{a}</Text>
                                </TouchableOpacity>
                            );
                        })}
                    </View>
                    <Text style={[S.label, { marginTop: Spacing.md }]}>Goal description</Text>
                    <TextInput
                        style={S.input}
                        value={newGoalText}
                        onChangeText={setNewGoalText}
                        placeholder={
                            newGoalActivity
                                ? `e.g. Nail my ${newGoalActivity} follow-through`
                                : 'Describe your goal'
                        }
                        placeholderTextColor={Colors.textMuted}
                        returnKeyType="done"
                        onSubmitEditing={handleAddGoal}
                        autoFocus
                    />
                    <TouchableOpacity
                        style={[S.cta, (!newGoalText.trim() || !newGoalActivity) && S.ctaDisabled]}
                        onPress={handleAddGoal}
                        disabled={!newGoalText.trim() || !newGoalActivity}
                        activeOpacity={0.85}
                    >
                        <Text style={[S.ctaText, (!newGoalText.trim() || !newGoalActivity) && S.ctaTextDisabled]}>
                            Add Goal
                        </Text>
                    </TouchableOpacity>
                </View>
            )}

            {/* Empty state */}
            {goals.length === 0 && !addingGoal && (
                <Text style={S.empty}>
                    No goals yet. Tap "+ New Goal" to start tracking your progress.
                </Text>
            )}

            {/* Goal List */}
            {goals.map(goal => {
                const latestRun = goal.runs[goal.runs.length - 1];
                const hasDelta =
                    latestRun?.improvement_delta !== null &&
                    latestRun?.improvement_delta !== undefined;
                const positive = (latestRun?.improvement_delta ?? 0) > 0;
                return (
                    <TouchableOpacity
                        key={goal.id}
                        style={S.goalCard}
                        onPress={() => navigation.navigate('GoalDetail', { goalId: goal.id })}
                        activeOpacity={0.75}
                    >
                        <View style={S.goalLeft}>
                            <Text style={S.goalActivity}>{goal.activityType}</Text>
                            <Text style={S.goalName} numberOfLines={2}>{goal.name}</Text>
                            <Text style={S.goalMeta}>
                                {goal.runs.length} session{goal.runs.length !== 1 ? 's' : ''}
                                {latestRun ? ` · Last score: ${latestRun.score}` : ''}
                            </Text>
                        </View>
                        <View style={S.goalRight}>
                            {latestRun ? (
                                <Text style={S.goalScore}>{latestRun.score}</Text>
                            ) : (
                                <Text style={S.goalNoScore}>—</Text>
                            )}
                            {hasDelta && (
                                <Text style={[S.goalDelta, { color: positive ? Colors.success : Colors.error }]}>
                                    {positive ? '▲' : '▼'} {Math.abs(latestRun!.improvement_delta!)}
                                </Text>
                            )}
                            <Text style={S.goalArrow}>›</Text>
                        </View>
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
    addBtn: {
        paddingHorizontal: 14,
        paddingVertical: 7,
        borderRadius: Radius.full,
        borderWidth: 1,
        borderColor: Colors.primaryBorder,
        backgroundColor: Colors.primaryDim,
    },
    addBtnText: {
        color: Colors.primary,
        fontSize: 13,
        fontWeight: '700',
    },
    card: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        padding: Spacing.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        marginBottom: Spacing.lg,
    },
    label: {
        fontSize: 11,
        fontWeight: '700',
        color: Colors.textSecondary,
        textTransform: 'uppercase',
        letterSpacing: 1.2,
        marginBottom: Spacing.sm,
    },
    chips: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, marginBottom: Spacing.sm },
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
    chipLabel: { color: Colors.textSecondary, fontSize: 13, fontWeight: '500' },
    chipLabelActive: { color: Colors.primary, fontWeight: '700' },
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
    cta: {
        backgroundColor: Colors.primary,
        borderRadius: Radius.lg,
        paddingVertical: 16,
        alignItems: 'center',
    },
    ctaDisabled: {
        backgroundColor: Colors.surface,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
    },
    ctaText: { color: Colors.background, fontSize: 15, fontWeight: '700' },
    ctaTextDisabled: { color: Colors.textMuted },
    empty: {
        color: Colors.textMuted,
        fontSize: 14,
        textAlign: 'center',
        marginTop: Spacing.xl,
        lineHeight: 22,
    },
    goalCard: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        borderLeftWidth: 3,
        borderLeftColor: Colors.primary,
        padding: Spacing.md,
        marginBottom: Spacing.sm,
        flexDirection: 'row',
        alignItems: 'center',
        overflow: 'hidden',
    },
    goalLeft: { flex: 1, marginRight: Spacing.sm },
    goalActivity: {
        color: Colors.primary,
        fontSize: 10,
        fontWeight: '700',
        textTransform: 'uppercase',
        letterSpacing: 1.2,
        marginBottom: 3,
    },
    goalName: { color: Colors.text, fontSize: 15, fontWeight: '700', marginBottom: 4 },
    goalMeta: { color: Colors.textMuted, fontSize: 12 },
    goalRight: { alignItems: 'flex-end' },
    goalScore: { color: Colors.primary, fontSize: 24, fontWeight: '800' },
    goalNoScore: { color: Colors.textMuted, fontSize: 22, fontWeight: '800' },
    goalDelta: { fontSize: 12, fontWeight: '700' },
    goalArrow: { color: Colors.textSecondary, fontSize: 20, marginTop: 2 },
});
