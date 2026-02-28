import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView, Dimensions, Alert } from 'react-native';
import { CameraView, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';

const screenWidth = Dimensions.get('window').width;
const cameraHeight = screenWidth * (16 / 9);

const ACTIVITIES = [
    { id: 'basketball_shot', label: '🏀 Basketball Shot' },
    { id: 'golf_swing', label: '⛳ Golf Swing' },
    { id: 'tennis_serve', label: '🎾 Tennis Serve' },
    { id: 'baseball_pitch', label: '⚾ Baseball Pitch' },
    { id: 'dance_move', label: '🕺 Dance Move' },
];

export default function RecordingScreen({ navigation }: any) {
    const [cameraPermission, requestCameraPermission] = useCameraPermissions();
    const [micPermission, requestMicPermission] = useMicrophonePermissions();

    const [isRecording, setIsRecording] = useState(false);
    const [timeLeft, setTimeLeft] = useState(5);
    const [activity, setActivity] = useState(ACTIVITIES[0].id);

    const cameraRef = useRef<CameraView>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        if (!cameraPermission?.granted) requestCameraPermission();
        if (!micPermission?.granted) requestMicPermission();
    }, [cameraPermission, micPermission]);

    useEffect(() => {
        if (isRecording && timeLeft > 0) {
            timerRef.current = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
        } else if (isRecording && timeLeft === 0) {
            stopRecording();
        }
        return () => {
            if (timerRef.current) clearTimeout(timerRef.current);
        };
    }, [isRecording, timeLeft]);

    const startRecording = async () => {
        if (!cameraRef.current) return;
        setIsRecording(true);
        setTimeLeft(5);
        try {
            const video = await cameraRef.current.recordAsync({ maxDuration: 5 });
            if (video && video.uri) {
                handleUpload(video.uri);
            }
        } catch (e) {
            console.error(e);
            setIsRecording(false);
            Alert.alert("Error", "Failed to record video.");
        }
    };

    const stopRecording = () => {
        if (cameraRef.current) {
            cameraRef.current.stopRecording();
        }
    };

    const handleUpload = (fileUri: string) => {
        setIsRecording(false);
        const metadata = {
            activity_type: activity,
            user_description: 'I want to improve my form',
        };
        // Navigate to Analyzing screen with the uri and metadata
        navigation.replace('Analyzing', { fileUri, metadata });
    };

    if (!cameraPermission?.granted || !micPermission?.granted) {
        return (
            <SafeAreaView style={styles.container}>
                <Text style={styles.permissionText}>We need camera and mic permissions to coach you.</Text>
                <TouchableOpacity style={styles.recordBtn} onPress={() => { requestCameraPermission(); requestMicPermission(); }}>
                    <Text style={styles.recordBtnText}>Grant Permissions</Text>
                </TouchableOpacity>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>Record Attempt</Text>
                <Text style={styles.subtitle}>Keep the subject fully in frame.</Text>
            </View>

            {/* Camera View locked to 16:9 */}
            <View style={styles.cameraContainer}>
                <CameraView
                    ref={cameraRef}
                    style={styles.camera}
                    mode="video"
                />

                {/* Visual Overlays for Recording */}
                {isRecording && (
                    <View style={styles.recordingOverlay}>
                        <View style={styles.recDot} />
                        <Text style={styles.timeText}>00:0{timeLeft}</Text>
                        <Text style={styles.warningText}>⚠️ HOLD PERFECTLY STILL</Text>
                    </View>
                )}
            </View>

            {/* Controls */}
            <View style={styles.controls}>
                <View style={styles.pickerContainer}>
                    {ACTIVITIES.map(act => (
                        <TouchableOpacity
                            key={act.id}
                            style={[styles.pickerItem, activity === act.id && styles.pickerItemActive]}
                            onPress={() => !isRecording && setActivity(act.id)}
                        >
                            <Text style={[styles.pickerText, activity === act.id && styles.pickerTextActive]}>
                                {act.label}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>

                {!isRecording ? (
                    <TouchableOpacity style={styles.recordBtn} onPress={startRecording}>
                        <View style={styles.recordBtnInner} />
                    </TouchableOpacity>
                ) : (
                    <TouchableOpacity style={styles.stopBtn} onPress={stopRecording}>
                        <View style={styles.stopBtnInner} />
                    </TouchableOpacity>
                )}
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#09090b' },
    header: { padding: 20, alignItems: 'center' },
    title: { fontSize: 24, fontWeight: '800', color: '#fff', marginBottom: 4 },
    subtitle: { fontSize: 14, color: '#a1a1aa', fontWeight: '500' },
    permissionText: { color: '#fff', textAlign: 'center', marginTop: 100, marginBottom: 20 },

    cameraContainer: {
        width: screenWidth,
        height: cameraHeight,
        backgroundColor: '#18181b',
        overflow: 'hidden',
        position: 'relative',
    },
    camera: { flex: 1 },
    recordingOverlay: {
        ...StyleSheet.absoluteFillObject,
        borderWidth: 3,
        borderColor: '#ef4444',
        alignItems: 'center',
        justifyContent: 'flex-start',
        paddingTop: 20,
    },
    recDot: { width: 12, height: 12, borderRadius: 6, backgroundColor: '#ef4444', position: 'absolute', top: 26, left: 20 },
    timeText: { color: '#ef4444', fontSize: 24, fontWeight: '800', textShadowColor: '#000', textShadowRadius: 4 },
    warningText: { color: '#fbbf24', fontSize: 16, fontWeight: '800', marginTop: 10, textShadowColor: '#000', textShadowRadius: 4 },

    controls: { flex: 1, justifyContent: 'space-between', paddingBottom: 40, paddingTop: 20 },
    pickerContainer: { flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'center', gap: 8, paddingHorizontal: 20 },
    pickerItem: { paddingVertical: 8, paddingHorizontal: 16, borderRadius: 100, backgroundColor: '#27272a', borderWidth: 1, borderColor: '#3f3f46' },
    pickerItemActive: { backgroundColor: '#3b82f6', borderColor: '#60a5fa' },
    pickerText: { color: '#a1a1aa', fontSize: 14, fontWeight: '600' },
    pickerTextActive: { color: '#ffffff' },

    recordBtn: { alignSelf: 'center', width: 72, height: 72, borderRadius: 36, borderWidth: 4, borderColor: '#ffffff', justifyContent: 'center', alignItems: 'center', backgroundColor: '#ef4444' },
    recordBtnInner: { width: 60, height: 60, borderRadius: 30, backgroundColor: '#ef4444' },
    stopBtn: { alignSelf: 'center', width: 72, height: 72, borderRadius: 36, borderWidth: 4, borderColor: '#ffffff', justifyContent: 'center', alignItems: 'center' },
    stopBtnInner: { width: 28, height: 28, borderRadius: 4, backgroundColor: '#ef4444' },
});
