import React, { useRef, useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus, Audio } from 'expo-av';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles, Colors } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Playback'>;
type PlaybackRouteProp = RouteProp<RootStackParamList, 'Playback'>;

export default function PlaybackScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<PlaybackRouteProp>();
    const { videoUri, data } = route.params;

    // Refs for AV control
    const videoRef = useRef<Video>(null);
    const soundRef = useRef<Audio.Sound | null>(null);

    // Sync Engine State
    const [isPausedForFeedback, setIsPausedForFeedback] = useState(false);
    const [hasShownFeedback, setHasShownFeedback] = useState(false);
    const [isVideoFinished, setIsVideoFinished] = useState(false);

    // Setup ElevenLabs Audio Proxy (Using dummy MP3 for now)
    useEffect(() => {
        let soundObject: Audio.Sound;

        const prepareAudio = async () => {
            try {
                await Audio.setAudioModeAsync({
                    playsInSilentModeIOS: true,
                    shouldDuckAndroid: false
                });

                const { sound } = await Audio.Sound.createAsync(
                    { uri: data.audio_url },
                    { shouldPlay: false }
                );

                soundRef.current = sound;
            } catch (e) {
                console.error("Failed to load ElevenLabs Audio:", e);
            }
        };

        prepareAudio();

        return () => {
            if (soundRef.current) {
                soundRef.current.unloadAsync();
            }
        };
    }, [data.audio_url]);

    // The Sync Engine
    const handlePlaybackStatusUpdate = (status: AVPlaybackStatus) => {
        if (!status.isLoaded) return;

        // 1. Trigger the AI Coaching Moment (Mistake Timestamp)
        if (!hasShownFeedback && status.positionMillis >= data.analysis.mistake_timestamp_ms) {
            if (videoRef.current) {
                videoRef.current.pauseAsync();
            }
            setIsPausedForFeedback(true);
            setHasShownFeedback(true);

            if (soundRef.current) {
                soundRef.current.playAsync(); // Start playing the ElevenLabs voice
            }
        }

        // 2. Video finished naturally
        if (status.didJustFinish) {
            setIsVideoFinished(true);
        }
    };

    const handleContinue = () => {
        setIsPausedForFeedback(false);
        if (soundRef.current) {
            soundRef.current.stopAsync(); // Stop narrating if user taps continue early
        }
        if (videoRef.current) {
            videoRef.current.playAsync(); // Resume normal video
        }
    };

    const finishSession = () => {
        navigation.replace('Complete', { data });
    };

    return (
        <View style={globalStyles.fullScreen}>

            {/* 16:9 Video Canvas exactly matching Camera Recording dimensions */}
            <View style={S.videoContainer}>
                <Video
                    ref={videoRef}
                    style={S.videoPlayer}
                    source={{ uri: videoUri }}
                    resizeMode={ResizeMode.COVER}
                    shouldPlay={true}
                    isMuted={true} // MUST be true so phone background noise doesn't drown out AI voice
                    progressUpdateIntervalMillis={50} // CRITICAL for accurate frame pausing
                    onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
                />

                {/* The SVG & Coaching Overlay */}
                {isPausedForFeedback && (
                    <View style={S.coachingOverlay}>
                        <View style={S.overlayBox}>
                            <Text style={S.coachingHeader}>Coach says:</Text>
                            <Text style={S.coachingText}>{data.analysis.coaching_script}</Text>
                            <Text style={S.timestampText}>Paused at {data.analysis.mistake_timestamp_ms}ms</Text>

                            <TouchableOpacity
                                style={[globalStyles.primaryButton, { marginTop: 20 }]}
                                onPress={handleContinue}
                            >
                                <Text style={globalStyles.buttonText}>Got it, Continue</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                )}
            </View>

            {/* Bottom Controls / Complete */}
            <View style={S.bottomControls}>
                <Text style={globalStyles.subHeading}>Playback Engine Running</Text>

                {isVideoFinished && (
                    <TouchableOpacity
                        style={globalStyles.primaryButton}
                        onPress={finishSession}
                    >
                        <Text style={globalStyles.buttonText}>View Session Results</Text>
                    </TouchableOpacity>
                )}
            </View>

        </View>
    );
}

const S = StyleSheet.create({
    videoContainer: {
        width: '100%',
        aspectRatio: 9 / 16, // Matches Camera precisely
        backgroundColor: '#000',
        overflow: 'hidden',
    },
    videoPlayer: {
        ...StyleSheet.absoluteFillObject,
    },
    coachingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.6)', // Dim the video heavily
        justifyContent: 'center',
        alignItems: 'center',
        padding: 24,
        zIndex: 10,
    },
    overlayBox: {
        backgroundColor: Colors.surface,
        padding: 24,
        borderRadius: 16,
        width: '100%',
        borderWidth: 2,
        borderColor: Colors.primary,
    },
    coachingHeader: {
        color: Colors.primary,
        fontSize: 16,
        fontWeight: 'bold',
        marginBottom: 8,
        textTransform: 'uppercase',
    },
    coachingText: {
        color: Colors.text,
        fontSize: 20,
        lineHeight: 28,
    },
    timestampText: {
        color: Colors.textSecondary,
        fontSize: 14,
        marginTop: 12,
    },
    bottomControls: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 24,
    }
});
