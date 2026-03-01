import React, { useState, useRef, useEffect } from "react";
import { View, Text, TouchableOpacity, StyleSheet } from "react-native";
import {
    CameraView,
    useCameraPermissions,
    useMicrophonePermissions,
} from "expo-camera";
import { useNavigation, useRoute, RouteProp } from "@react-navigation/native";
import { NativeStackNavigationProp } from "@react-navigation/native-stack";
import { RootStackParamList } from "../types";
import { globalStyles, Colors } from "../styles/theme";

type NavigationProp = NativeStackNavigationProp<
    RootStackParamList,
    "Recording"
>;
type RecordingRouteProp = RouteProp<RootStackParamList, "Recording">;

export default function RecordingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<RecordingRouteProp>();
    const {
        activityType,
        previousData,
        description = `I want to improve my ${activityType} form.`,
        goalId,
        voiceId,
    } = route.params;

    const [cameraPermission, requestCameraPermission] = useCameraPermissions();
    const [micPermission, requestMicPermission] = useMicrophonePermissions();

    const cameraRef = useRef<CameraView>(null);
    const [isReady, setIsReady] = useState(false);
    const [isPreCountdown, setIsPreCountdown] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [preCountdown, setPreCountdown] = useState(0); // 3-second pre-recording countdown
    const [recordingCountdown, setRecordingCountdown] = useState(5); // 5-second recording countdown
    const [facing, setFacing] = useState<"front" | "back">("back");

    useEffect(() => {
        if (!isPreCountdown || preCountdown <= 0) return;

        const timer = setTimeout(() => setPreCountdown(c => c - 1), 1000);
        return () => clearTimeout(timer);
    }, [isPreCountdown, preCountdown]);

    useEffect(() => {
        if (!isPreCountdown || preCountdown !== 0) return;

        setIsPreCountdown(false);
        setIsRecording(true);
        void actuallyStartRecording();
    }, [isPreCountdown, preCountdown]);

    useEffect(() => {
        if (!isRecording || recordingCountdown <= 0) return;

        const timer = setTimeout(() => setRecordingCountdown(c => c - 1), 1000);
        return () => clearTimeout(timer);
    }, [isRecording, recordingCountdown]);

    useEffect(() => {
        if (isRecording && recordingCountdown === 0) {
            stopRecording();
        }
    }, [isRecording, recordingCountdown]);

    if (!cameraPermission || !micPermission) return <View />;

    if (!cameraPermission.granted || !micPermission.granted) {
        return (
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>Permissions Required</Text>
                <Text style={globalStyles.subHeading}>
                    We need camera and microphone to analyze your form.
                </Text>
                <TouchableOpacity
                    style={globalStyles.primaryButton}
                    onPress={() => {
                        requestCameraPermission();
                        requestMicPermission();
                    }}
                >
                    <Text style={globalStyles.buttonText}>
                        Grant Permissions
                    </Text>
                </TouchableOpacity>
            </View>
        );
    }

    const startRecording = async () => {
        if (!cameraRef.current || !isReady || isPreCountdown || isRecording)
            return;
        setIsPreCountdown(true);
        setPreCountdown(3); // Start 3-second pre-recording countdown
        setRecordingCountdown(5);
    };

    const actuallyStartRecording = async () => {
        if (!cameraRef.current) return;

        try {
            const video = await cameraRef.current.recordAsync({
                maxDuration: 5,
            });
            if (video?.uri) {
                navigation.replace("Analyzing", {
                    videoUri: video.uri,
                    activityType,
                    description,
                    previousData,
                    goalId,
                    voiceId,
                });
            }
        } catch (error) {
            console.error("Recording failed:", error);
        } finally {
            setIsRecording(false);
        }
    };

    const stopRecording = () => {
        if (cameraRef.current && isRecording) {
            cameraRef.current.stopRecording();
            setIsRecording(false);
        }
    };

    return (
        <View style={S.screen}>
            {/* Camera fills entire screen */}
            <CameraView
                ref={cameraRef}
                style={StyleSheet.absoluteFill}
                facing={facing}
                mode="video"
                onCameraReady={() => setIsReady(true)}
            />

            {/* Countdown overlay */}
            {(isPreCountdown || (isRecording && recordingCountdown > 0)) && (
                <View style={S.recordingRing} pointerEvents="none">
                    <Text style={S.countdownText}>
                        {isPreCountdown ? preCountdown : recordingCountdown}
                    </Text>
                    <Text style={S.warningText}>
                        {isPreCountdown
                            ? "Get ready!"
                            : "Keep the phone still!"}
                    </Text>
                </View>
            )}

            {/* Top badge */}
            <View style={S.topBar}>
                <Text style={S.activityBadge}>{activityType}</Text>
            </View>

            {/* Bottom controls bar */}
            <View style={S.bottomBar}>
                {!isRecording && !isPreCountdown ? (
                    <TouchableOpacity
                        style={S.flipBtn}
                        onPress={() =>
                            setFacing(f => (f === "back" ? "front" : "back"))
                        }
                    >
                        <Text style={S.flipBtnText}>↺ Flip</Text>
                    </TouchableOpacity>
                ) : isPreCountdown ? (
                    // During pre-count, show cancel button
                    <TouchableOpacity
                        style={[
                            S.flipBtn,
                            { backgroundColor: "rgba(255,100,100,0.3)" },
                        ]}
                        onPress={() => {
                            setIsPreCountdown(false);
                            setIsRecording(false);
                            setPreCountdown(0);
                            setRecordingCountdown(5);
                        }}
                    >
                        <Text style={S.flipBtnText}>Cancel</Text>
                    </TouchableOpacity>
                ) : (
                    <View style={S.placeholder} />
                )}

                <TouchableOpacity
                    style={[
                        S.recordBtn,
                        (isRecording || isPreCountdown) && S.recordBtnActive,
                    ]}
                    onPress={isRecording ? stopRecording : startRecording}
                    disabled={!isReady || isPreCountdown}
                >
                    <View
                        style={
                            isRecording && recordingCountdown > 0
                                ? S.recordInnerSquare
                                : S.recordInnerCircle
                        }
                    />
                </TouchableOpacity>

                <View style={S.placeholder} />
            </View>
        </View>
    );
}

const S = StyleSheet.create({
    screen: {
        flex: 1,
        backgroundColor: "#000",
    },
    topBar: {
        position: "absolute",
        top: 56,
        left: 0,
        right: 0,
        alignItems: "center",
    },
    activityBadge: {
        color: "#fff",
        fontSize: 15,
        fontWeight: "700",
        backgroundColor: "rgba(0,0,0,0.5)",
        paddingHorizontal: 16,
        paddingVertical: 6,
        borderRadius: 20,
        overflow: "hidden",
    },
    bottomBar: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        height: 160,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingHorizontal: 40,
        paddingBottom: 40, // clears home indicator
        backgroundColor: "rgba(0,0,0,0.45)",
    },
    recordBtn: {
        width: 76,
        height: 76,
        borderRadius: 38,
        borderWidth: 4,
        borderColor: "#fff",
        alignItems: "center",
        justifyContent: "center",
    },
    recordBtnActive: {
        borderColor: Colors.error,
    },
    recordInnerCircle: {
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: Colors.error,
    },
    recordInnerSquare: {
        width: 28,
        height: 28,
        borderRadius: 4,
        backgroundColor: Colors.error,
    },
    flipBtn: {
        backgroundColor: "rgba(255,255,255,0.15)",
        borderRadius: 20,
        paddingHorizontal: 16,
        paddingVertical: 9,
    },
    flipBtnText: {
        color: "#fff",
        fontSize: 14,
        fontWeight: "600",
    },
    placeholder: {
        width: 70,
    },
    recordingRing: {
        ...StyleSheet.absoluteFillObject,
        borderWidth: 5,
        borderColor: Colors.error,
        justifyContent: "center",
        alignItems: "center",
    },
    countdownText: {
        fontSize: 96,
        fontWeight: "bold",
        color: "#fff",
        textShadowColor: "rgba(0,0,0,0.8)",
        textShadowOffset: { width: 0, height: 2 },
        textShadowRadius: 12,
    },
    warningText: {
        fontSize: 20,
        fontWeight: "bold",
        color: Colors.error,
        marginTop: 12,
        backgroundColor: "rgba(0,0,0,0.65)",
        paddingHorizontal: 16,
        paddingVertical: 6,
        borderRadius: 8,
    },
});
