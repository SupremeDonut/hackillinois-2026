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

    const videoRef = useRef<Video>(null);
    const soundRef = useRef<Audio.Sound | null>(null);

    // Sync Engine State
    const [isPausedForFeedback, setIsPausedForFeedback] = useState(false);
    const [showingFrame, setShowingFrame] = useState(false); // Controls the two-step UI
    const [currentFeedbackIndex, setCurrentFeedbackIndex] = useState(0);
    const [isVideoFinished, setIsVideoFinished] = useState(false);

    // Derive the current target based on index
    const currentFeedback = data.feedback_points?.[currentFeedbackIndex];

    // Load ElevenLabs Audio for the CURRENT feedback point
    useEffect(() => {
        let soundObject: Audio.Sound;

        const prepareAudio = async () => {
            // Clean up previous audio if it exists
            if (soundRef.current) {
                await soundRef.current.unloadAsync();
                soundRef.current = null;
            }

            if (!currentFeedback?.audio_url) return;

            try {
                await Audio.setAudioModeAsync({
                    playsInSilentModeIOS: true,
                    shouldDuckAndroid: false
                });

                const { sound } = await Audio.Sound.createAsync(
                    { uri: currentFeedback.audio_url },
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
    }, [currentFeedbackIndex, currentFeedback?.audio_url]);

    // The Sync Engine
    const handlePlaybackStatusUpdate = (status: AVPlaybackStatus) => {
        if (!status.isLoaded) return;

        // 1. Trigger the specific Coaching Moment
        if (currentFeedback && !isPausedForFeedback && status.positionMillis >= currentFeedback.mistake_timestamp_ms) {
            if (videoRef.current) {
                videoRef.current.pauseAsync();
            }
            setIsPausedForFeedback(true);
            setShowingFrame(false); // Make sure we start with the text box, not the frame

            if (soundRef.current) {
                soundRef.current.playAsync(); // Start playing voice for this specific instruction
            }
        }

        // 2. Video finished naturally
        if (status.didJustFinish) {
            setIsVideoFinished(true);
        }
    };

    const handleShowFrame = () => {
        // Step 1 -> Step 2 transition
        setShowingFrame(true);
    };

    const handleContinue = () => {
        // End of Step 2 -> Resume Video
        setIsPausedForFeedback(false);
        setShowingFrame(false);

        if (soundRef.current) {
            soundRef.current.stopAsync();
        }

        setCurrentFeedbackIndex(prev => prev + 1);

        if (videoRef.current) {
            videoRef.current.playAsync();
        }
    };

    const finishSession = () => {
        navigation.replace('Complete', { data });
    };

    return (
        <View style={globalStyles.fullScreen}>

            <View style={S.videoContainer}>
                <Video
                    ref={videoRef}
                    style={S.videoPlayer}
                    source={{ uri: videoUri }}
                    resizeMode={ResizeMode.COVER}
                    shouldPlay={true}
                    isMuted={true}
                    progressUpdateIntervalMillis={50}
                    onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
                />

                {isPausedForFeedback && currentFeedback && (
                    <View style={showingFrame ? S.transparentOverlay : S.coachingOverlay}>

                        {/* STEP 1: The Text Box Prompt */}
                        {!showingFrame && (
                            <View style={S.overlayBox}>
                                <Text style={S.coachingHeader}>Coach says:</Text>
                                <Text style={S.coachingText}>{currentFeedback.coaching_script}</Text>
                                <Text style={S.timestampText}>Paused at {currentFeedback.mistake_timestamp_ms}ms</Text>

                                <TouchableOpacity
                                    style={[globalStyles.primaryButton, { marginTop: 20 }]}
                                    onPress={handleShowFrame}
                                >
                                    <Text style={globalStyles.buttonText}>Show Corrected Frame</Text>
                                </TouchableOpacity>
                            </View>
                        )}

                        {/* STEP 2: The Visible Frame + SVG Overlay (Dev 3 adds SVG here later) + Continue Button */}
                        {showingFrame && (
                            <View style={S.frameOverlayContainer}>
                                {/* Dev 3 SVG overlays will mount naturally into this parent canvas later */}

                                <TouchableOpacity
                                    style={[globalStyles.primaryButton, S.floatingContinueButton]}
                                    onPress={handleContinue}
                                >
                                    <Text style={globalStyles.buttonText}>
                                        {currentFeedbackIndex < (data.feedback_points?.length || 0) - 1 ? 'Next' : 'Continue'}
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        )}

                    </View>
                )}
            </View>

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
        flex: 1,
        width: '100%',
        aspectRatio: 9 / 16,
        backgroundColor: '#000',
        overflow: 'hidden',
    },
    videoPlayer: {
        ...StyleSheet.absoluteFillObject,
    },
    coachingOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.6)',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 24,
        zIndex: 10,
    },
    transparentOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'transparent',
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
    frameOverlayContainer: {
        ...StyleSheet.absoluteFillObject,
        justifyContent: 'flex-end',
        padding: 24,
    },
    floatingContinueButton: {
        backgroundColor: Colors.primary,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 4,
        elevation: 8,
    },
    bottomControls: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 24,
    }
});
