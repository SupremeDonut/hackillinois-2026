import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, ActivityType } from '../types';
import { globalStyles, Colors } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Recording'>;
type RecordingRouteProp = RouteProp<RootStackParamList, 'Recording'>;

export default function RecordingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<RecordingRouteProp>();
    const { activityType, previousData } = route.params;

    const [cameraPermission, requestCameraPermission] = useCameraPermissions();
    const [micPermission, requestMicPermission] = useMicrophonePermissions();

    const cameraRef = useRef<CameraView>(null);
    const [isReady, setIsReady] = useState(false);
    const [isRecording, setIsRecording] = useState(false);
    const [facing, setFacing] = useState<'front' | 'back'>('back');

    const [description, setDescription] = useState('My default description for Stage 2');
    const [countdown, setCountdown] = useState(5);

    useEffect(() => {
        let timer: NodeJS.Timeout;
        if (isRecording && countdown > 0) {
            timer = setTimeout(() => setCountdown(c => c - 1), 1000);
        } else if (isRecording && countdown === 0) {
            stopRecording();
        }
        return () => clearTimeout(timer);
    }, [isRecording, countdown]);

    if (!cameraPermission || !micPermission) {
        return <View />; // Loading permissions
    }

    if (!cameraPermission.granted || !micPermission.granted) {
        return (
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>Permissions Required</Text>
                <Text style={globalStyles.subHeading}>We need camera and microphone to analyze your form.</Text>
                <TouchableOpacity style={globalStyles.primaryButton} onPress={() => {
                    requestCameraPermission();
                    requestMicPermission();
                }}>
                    <Text style={globalStyles.buttonText}>Grant Permissions</Text>
                </TouchableOpacity>
            </View>
        );
    }

    const startRecording = async () => {
        if (!cameraRef.current || !isReady) return;

        setIsRecording(true);
        setCountdown(5);

        try {
            const video = await cameraRef.current.recordAsync({ maxDuration: 5 });

            console.log('Video saved locally to:', video?.uri);

            if (video?.uri) {
                navigation.replace('Analyzing', {
                    videoUri: video.uri,
                    activityType,
                    description,
                    previousData
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
        <View style={globalStyles.fullScreen}>

            {/* 16:9 Aspect Ratio Camera Container */}
            <View style={S.cameraContainer}>
                <CameraView
                    ref={cameraRef}
                    style={S.camera}
                    facing={facing}
                    mode="video"
                    onCameraReady={() => setIsReady(true)}
                />

                {/* Flip button — hidden while recording */}
                {!isRecording && (
                    <TouchableOpacity
                        style={S.flipButton}
                        onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
                    >
                        <Text style={S.flipButtonText}>↺ Flip</Text>
                    </TouchableOpacity>
                )}

                {/* Overlay positioned absolutely over the CameraView */}
                {isRecording && (
                    <View style={S.recordingOverlay} pointerEvents="none">
                        <Text style={S.countdownText}>{countdown}s</Text>
                        <Text style={S.warningText}>Keep the phone absolutely still!</Text>
                    </View>
                )}
            </View>

            {/* Controls Container */}
            <View style={S.controlsContainer}>
                <Text style={globalStyles.subHeading}>Activity: {activityType}</Text>

                <TouchableOpacity
                    style={[globalStyles.primaryButton, isRecording && S.stopButton]}
                    onPress={isRecording ? stopRecording : startRecording}
                    disabled={!isReady}
                >
                    <Text style={globalStyles.buttonText}>
                        {!isReady ? 'Loading Camera...' : (isRecording ? 'Stop' : 'Start Recording')}
                    </Text>
                </TouchableOpacity>
            </View>

        </View>
    );
}

const S = StyleSheet.create({
    cameraContainer: {
        width: '100%',
        aspectRatio: 9 / 16,
        backgroundColor: '#000',
        overflow: 'hidden',
    },
    camera: {
        flex: 1,
    },
    recordingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(255,0,0,0.1)',
        borderWidth: 4,
        borderColor: Colors.error,
        justifyContent: 'center',
        alignItems: 'center',
    },
    countdownText: {
        fontSize: 72,
        fontWeight: 'bold',
        color: Colors.text,
        textShadowColor: 'rgba(0, 0, 0, 0.75)',
        textShadowOffset: { width: -1, height: 1 },
        textShadowRadius: 10
    },
    warningText: {
        fontSize: 24,
        fontWeight: 'bold',
        color: Colors.error,
        marginTop: 20,
        backgroundColor: 'rgba(0,0,0,0.7)',
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 8,
    },
    controlsContainer: {
        flex: 1,
        padding: 24,
        justifyContent: 'center',
        alignItems: 'center',
    },
    stopButton: {
        backgroundColor: Colors.error,
    },
    flipButton: {
        position: 'absolute',
        top: 12,
        right: 12,
        backgroundColor: 'rgba(0,0,0,0.55)',
        borderRadius: 20,
        paddingHorizontal: 14,
        paddingVertical: 7,
    },
    flipButtonText: {
        color: Colors.text,
        fontSize: 14,
        fontWeight: '600',
    },
});
