import React, { useEffect, useRef } from 'react';
import { View, Text, Animated, StyleSheet } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, AnalysisResponse } from '../types';
import { Colors, Spacing, Radius } from '../styles/theme';
import { uploadVideo } from '../services/api';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Analyzing'>;
type AnalyzingRouteProp = RouteProp<RootStackParamList, 'Analyzing'>;

export default function AnalyzingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<AnalyzingRouteProp>();
    const { videoUri, activityType, description, previousData } = route.params;

    // Two rotating arcs for a clean spinner feel
    const rotate1 = useRef(new Animated.Value(0)).current;
    const rotate2 = useRef(new Animated.Value(0)).current;
    const fade = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        Animated.timing(fade, { toValue: 1, duration: 400, useNativeDriver: true }).start();

        Animated.loop(
            Animated.timing(rotate1, { toValue: 1, duration: 1200, useNativeDriver: true })
        ).start();
        Animated.loop(
            Animated.timing(rotate2, { toValue: -1, duration: 1800, useNativeDriver: true })
        ).start();
    }, []);

    useEffect(() => {
        let cancelled = false;
        const runAnalysis = async () => {
            const data = await uploadVideo({
                videoUri, activityType, description, previousData,
                _useMockRetry: !!previousData,
            } as any);
            if (!cancelled) {
                navigation.replace('Playback', {
                    videoUri,
                    data: data as AnalysisResponse,
                    activityType,
                });
            }
        };
        runAnalysis();
        return () => { cancelled = true; };
    }, []);

    const spin1 = rotate1.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });
    const spin2 = rotate2.interpolate({ inputRange: [-1, 0], outputRange: ['-360deg', '0deg'] });

    return (
        <Animated.View style={[S.screen, { opacity: fade }]}>
            {/* Geometric spinner — two counter-rotating arcs */}
            <View style={S.spinnerContainer}>
                <Animated.View style={[S.arc, S.arcOuter, { transform: [{ rotate: spin1 }] }]} />
                <Animated.View style={[S.arc, S.arcInner, { transform: [{ rotate: spin2 }] }]} />
                <View style={S.dot} />
            </View>

            <Text style={S.title}>Reviewing your form</Text>
            <Text style={S.subtitle}>{activityType}</Text>
        </Animated.View>
    );
}

const S = StyleSheet.create({
    screen: {
        flex: 1,
        backgroundColor: Colors.background,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: Spacing.lg,
    },
    spinnerContainer: {
        width: 100,
        height: 100,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: Spacing.xl,
    },
    arc: {
        position: 'absolute',
        borderRadius: 999,
        borderColor: 'transparent',
    },
    arcOuter: {
        width: 100,
        height: 100,
        borderWidth: 3,
        borderTopColor: Colors.primary,
        borderRightColor: Colors.primaryDim,
    },
    arcInner: {
        width: 68,
        height: 68,
        borderWidth: 2.5,
        borderBottomColor: Colors.primary,
        borderLeftColor: Colors.primaryDim,
        opacity: 0.7,
    },
    dot: {
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: Colors.primary,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.8,
        shadowRadius: 8,
        elevation: 6,
    },
    title: {
        fontSize: 22,
        fontWeight: '700',
        color: Colors.text,
        letterSpacing: -0.4,
        marginBottom: 6,
    },
    subtitle: {
        fontSize: 13,
        color: Colors.textSecondary,
        fontWeight: '500',
        letterSpacing: 0.3,
    },
});
