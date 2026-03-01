import React, { useEffect, useRef, useState } from "react";
import { View, Text, Animated, StyleSheet } from "react-native";
import { useNavigation, useRoute, RouteProp } from "@react-navigation/native";
import { NativeStackNavigationProp } from "@react-navigation/native-stack";
import { RootStackParamList, AnalysisResponse } from "../types";
import { Colors, Spacing, Radius } from "../styles/theme";
import { uploadVideo } from "../services/api";

type NavigationProp = NativeStackNavigationProp<
    RootStackParamList,
    "Analyzing"
>;
type AnalyzingRouteProp = RouteProp<RootStackParamList, "Analyzing">;

const STAGES: { label: string; target: number; duration: number }[] = [
    { label: "Uploading video...", target: 0.08, duration: 800 },
    { label: "Detecting pose keypoints...", target: 0.22, duration: 3500 },
    { label: "Analysing movement with AI...", target: 0.55, duration: 28000 },
    { label: "Generating coaching feedback...", target: 0.72, duration: 8000 },
    { label: "Synthesising voice audio...", target: 0.88, duration: 6000 },
    { label: "Building skeleton overlays...", target: 0.95, duration: 4000 },
    { label: "Almost there...", target: 0.98, duration: 4000 },
];

const BAR_WIDTH = 280;

export default function AnalyzingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<AnalyzingRouteProp>();
    const {
        videoUri,
        activityType,
        voiceId,
        description,
        previousData,
        goalId,
    } = route.params;

    const rotate1 = useRef(new Animated.Value(0)).current;
    const rotate2 = useRef(new Animated.Value(0)).current;
    const fade = useRef(new Animated.Value(0)).current;
    const barAnim = useRef(new Animated.Value(0)).current;

    const [stageIndex, setStageIndex] = useState(0);
    const stageIndexRef = useRef(0);

    useEffect(() => {
        Animated.timing(fade, {
            toValue: 1,
            duration: 400,
            useNativeDriver: true,
        }).start();
        Animated.loop(
            Animated.timing(rotate1, {
                toValue: 1,
                duration: 1200,
                useNativeDriver: true,
            }),
        ).start();
        Animated.loop(
            Animated.timing(rotate2, {
                toValue: -1,
                duration: 1800,
                useNativeDriver: true,
            }),
        ).start();
    }, []);

    // Advance through stages sequentially
    useEffect(() => {
        let completed = false;
        function advanceStage(idx: number) {
            if (completed || idx >= STAGES.length) return;
            const stage = STAGES[idx];
            Animated.timing(barAnim, {
                toValue: stage.target * BAR_WIDTH,
                duration: stage.duration,
                useNativeDriver: false,
            }).start(({ finished }) => {
                if (!finished || completed) return;
                stageIndexRef.current = idx + 1;
                setStageIndex(idx + 1);
                advanceStage(idx + 1);
            });
        }
        advanceStage(0);
        return () => {
            completed = true;
        };
    }, []);

    // When API returns, snap bar to 100%
    const completeBar = () => {
        Animated.timing(barAnim, {
            toValue: BAR_WIDTH,
            duration: 400,
            useNativeDriver: false,
        }).start();
        setStageIndex(STAGES.length); // clear label
    };

    useEffect(() => {
        let cancelled = false;
        const runAnalysis = async () => {
            const data = await uploadVideo({
                videoUri,
                activityType,
                description,
                previousData,
                voiceId,
                _useMockRetry: !!previousData,
            } as any);
            if (!cancelled) {
                completeBar();
                // Short pause so user sees 100% before navigating
                setTimeout(() => {
                    if (!cancelled) {
                        navigation.replace("Playback", {
                            videoUri,
                            data: data as AnalysisResponse,
                            activityType,
                            goalId,
                            voiceId,
                        });
                    }
                }, 350);
            }
        };
        runAnalysis();
        return () => {
            cancelled = true;
        };
    }, []);

    const spin1 = rotate1.interpolate({
        inputRange: [0, 1],
        outputRange: ["0deg", "360deg"],
    });
    const spin2 = rotate2.interpolate({
        inputRange: [-1, 0],
        outputRange: ["-360deg", "0deg"],
    });

    const currentLabel =
        stageIndex < STAGES.length
            ? STAGES[stageIndex].label
            : "Finishing up...";

    return (
        <Animated.View style={[S.screen, { opacity: fade }]}>
            {/* Geometric spinner */}
            <View style={S.spinnerContainer}>
                <Animated.View
                    style={[
                        S.arc,
                        S.arcOuter,
                        { transform: [{ rotate: spin1 }] },
                    ]}
                />
                <Animated.View
                    style={[
                        S.arc,
                        S.arcInner,
                        { transform: [{ rotate: spin2 }] },
                    ]}
                />
                <View style={S.dot} />
            </View>

            <Text style={S.title}>Reviewing your form</Text>
            <Text style={S.subtitle}>{activityType}</Text>

            {/* Progress bar */}
            <View style={S.barTrack}>
                <Animated.View style={[S.barFill, { width: barAnim }]} />
            </View>
            <Text style={S.stageLabel}>{currentLabel}</Text>
        </Animated.View>
    );
}

const S = StyleSheet.create({
    screen: {
        flex: 1,
        backgroundColor: Colors.background,
        alignItems: "center",
        justifyContent: "center",
        paddingHorizontal: Spacing.lg,
    },
    spinnerContainer: {
        width: 100,
        height: 100,
        alignItems: "center",
        justifyContent: "center",
        marginBottom: Spacing.xl,
    },
    arc: {
        position: "absolute",
        borderRadius: 999,
        borderColor: "transparent",
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
        fontWeight: "700",
        color: Colors.text,
        letterSpacing: -0.4,
        marginBottom: 6,
    },
    subtitle: {
        fontSize: 13,
        color: Colors.textSecondary,
        fontWeight: "500",
        letterSpacing: 0.3,
        marginBottom: Spacing.xl,
    },
    barTrack: {
        width: BAR_WIDTH,
        height: 4,
        borderRadius: 2,
        backgroundColor: "rgba(255,255,255,0.08)",
        overflow: "hidden",
    },
    barFill: {
        height: 4,
        borderRadius: 2,
        backgroundColor: Colors.primary,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.8,
        shadowRadius: 6,
    },
    stageLabel: {
        marginTop: 12,
        fontSize: 13,
        color: Colors.textSecondary,
        fontWeight: "500",
        letterSpacing: 0.2,
    },
});
