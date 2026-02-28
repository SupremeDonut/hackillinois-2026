import React, { useState } from 'react';
import {
    View, Text, TouchableOpacity, StyleSheet, TextInput,
    KeyboardAvoidingView, Platform, ScrollView,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';

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
    const isReady = activityType.trim().length > 0;

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

                {/* ── CTA ── */}
                <TouchableOpacity
                    style={[S.cta, !isReady && S.ctaDisabled]}
                    onPress={() => isReady && navigation.navigate('Recording', { activityType })}
                    disabled={!isReady}
                    activeOpacity={0.85}
                >
                    <Text style={[S.ctaText, !isReady && S.ctaTextDisabled]}>
                        Start Recording
                    </Text>
                </TouchableOpacity>

                <Text style={S.hint}>Max 5 seconds of footage</Text>
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
    },
});
