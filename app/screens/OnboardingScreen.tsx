import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    SafeAreaView,
    KeyboardAvoidingView,
    Platform,
    ScrollView,
} from 'react-native';
import Svg, { Path, Ellipse } from 'react-native-svg';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { saveAccount } from '../services/accountStore';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Onboarding'>;

export default function OnboardingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const [displayName, setDisplayName] = useState('');
    const [email, setEmail] = useState('');
    const [error, setError] = useState('');

    const isValid = displayName.trim().length > 0 && email.trim().includes('@');

    const handleGetStarted = async () => {
        if (!displayName.trim()) {
            setError('Please enter your name.');
            return;
        }
        if (!email.trim().includes('@')) {
            setError('Please enter a valid email address.');
            return;
        }
        await saveAccount({
            displayName: displayName.trim(),
            email: email.trim().toLowerCase(),
            createdAt: new Date().toISOString(),
        });
        navigation.replace('Home');
    };

    return (
        <SafeAreaView style={S.safe}>
            <KeyboardAvoidingView
                behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                style={S.kav}
            >
                <ScrollView
                    contentContainerStyle={S.scroll}
                    keyboardShouldPersistTaps="handled"
                    showsVerticalScrollIndicator={false}
                >
                    {/* Brand */}
                    <View style={S.brand}>
                        <Svg width={88} height={72} viewBox="0 0 88 72" style={{ marginBottom: 14 }}>
                            <Path d="M44 36 C36 28, 14 20, 8 10 C4 2, 16 0, 24 6 C32 12, 40 24, 44 36Z" fill="#00E5A0" opacity={0.95} />
                            <Path d="M44 36 C52 28, 74 20, 80 10 C84 2, 72 0, 64 6 C56 12, 48 24, 44 36Z" fill="#00E5A0" opacity={0.95} />
                            <Path d="M44 38 C36 44, 12 46, 8 56 C4 64, 18 68, 28 60 C36 54, 42 46, 44 38Z" fill="#00C47A" opacity={0.85} />
                            <Path d="M44 38 C52 44, 76 46, 80 56 C84 64, 70 68, 60 60 C52 54, 46 46, 44 38Z" fill="#00C47A" opacity={0.85} />
                            <Ellipse cx={26} cy={18} rx={5} ry={3} fill="rgba(255,255,255,0.25)" />
                            <Ellipse cx={62} cy={18} rx={5} ry={3} fill="rgba(255,255,255,0.25)" />
                            <Ellipse cx={44} cy={37} rx={3} ry={14} fill="#E8FFF7" opacity={0.9} />
                            <Path d="M42 24 C40 16, 34 10, 30 6" stroke="#90EDB0" strokeWidth={1.5} fill="none" strokeLinecap="round" />
                            <Ellipse cx={30} cy={6} rx={2} ry={2} fill="#90EDB0" />
                            <Path d="M46 24 C48 16, 54 10, 58 6" stroke="#90EDB0" strokeWidth={1.5} fill="none" strokeLinecap="round" />
                            <Ellipse cx={58} cy={6} rx={2} ry={2} fill="#90EDB0" />
                        </Svg>
                        <Text style={S.appName}>Morphi</Text>
                        <Text style={S.tagline}>Learning for every hobby.</Text>
                    </View>

                    {/* Intro text */}
                    <Text style={S.welcomeTitle}>Create your account</Text>
                    <Text style={S.welcomeSub}>
                        Your profile is stored on this device. It helps personalize your coaching experience.
                    </Text>

                    {/* Form card */}
                    <View style={S.card}>
                        <Text style={S.inputLabel}>Your name</Text>
                        <TextInput
                            style={S.input}
                            value={displayName}
                            onChangeText={(t) => { setDisplayName(t); setError(''); }}
                            placeholder="e.g. Alex"
                            placeholderTextColor={Colors.textMuted}
                            autoCapitalize="words"
                            returnKeyType="next"
                        />
                        <Text style={S.inputLabel}>Email address</Text>
                        <TextInput
                            style={[S.input, { marginBottom: 0 }]}
                            value={email}
                            onChangeText={(t) => { setEmail(t); setError(''); }}
                            placeholder="e.g. alex@example.com"
                            placeholderTextColor={Colors.textMuted}
                            keyboardType="email-address"
                            autoCapitalize="none"
                            returnKeyType="done"
                            onSubmitEditing={handleGetStarted}
                        />
                    </View>

                    {error ? <Text style={S.errorText}>{error}</Text> : null}

                    <TouchableOpacity
                        style={[S.cta, !isValid && S.ctaDisabled]}
                        onPress={handleGetStarted}
                        disabled={!isValid}
                        activeOpacity={0.85}
                    >
                        <Text style={[S.ctaText, !isValid && S.ctaTextDisabled]}>
                            Get Started
                        </Text>
                    </TouchableOpacity>
                </ScrollView>
            </KeyboardAvoidingView>
        </SafeAreaView>
    );
}

const S = StyleSheet.create({
    safe: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    kav: {
        flex: 1,
    },
    scroll: {
        flexGrow: 1,
        paddingHorizontal: Spacing.lg,
        paddingTop: 48,
        paddingBottom: 48,
    },
    brand: {
        alignItems: 'center',
        marginBottom: 36,
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
    welcomeTitle: {
        fontSize: 26,
        fontWeight: '800',
        color: Colors.text,
        letterSpacing: -0.5,
        marginBottom: Spacing.sm,
    },
    welcomeSub: {
        fontSize: 14,
        color: Colors.textSecondary,
        lineHeight: 21,
        marginBottom: Spacing.xl,
    },
    card: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        padding: Spacing.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        marginBottom: Spacing.md,
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
    errorText: {
        color: Colors.error,
        fontSize: 13,
        marginBottom: Spacing.sm,
        textAlign: 'center',
    },
    cta: {
        backgroundColor: Colors.primary,
        borderRadius: Radius.lg,
        paddingVertical: 18,
        alignItems: 'center',
        marginTop: Spacing.sm,
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
});
