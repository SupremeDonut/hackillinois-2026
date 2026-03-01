import React, { useRef, useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Dimensions } from 'react-native';
import { Video, ResizeMode, AVPlaybackStatus, Audio } from 'expo-av';
import * as FileSystem from 'expo-file-system/legacy';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles, Colors } from '../styles/theme';
import SVGOverlay from '../components/SVGOverlay';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Playback'>;
type PlaybackRouteProp = RouteProp<RootStackParamList, 'Playback'>;

export default function PlaybackScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<PlaybackRouteProp>();
    const { videoUri, data, activityType, goalId } = route.params;

    const videoRef = useRef<Video>(null);
    const soundRef = useRef<Audio.Sound | null>(null);

    // Lock ref to prevent the high-frequency 50ms loop from triggering the same audio repeatedly point before React state updates
    const hasTriggeredCurrentFeedback = useRef(false);

    // Sync Engine State
    const [isPausedForFeedback, setIsPausedForFeedback] = useState(false);
    const [showingFrame, setShowingFrame] = useState(false); // Controls the two-step UI
    const [currentFeedbackIndex, setCurrentFeedbackIndex] = useState(0);
    const [isVideoFinished, setIsVideoFinished] = useState(false);

    // Layout Capture to pass absolute pixel boundaries to the SVG Overlay engine
    const [videoLayout, setVideoLayout] = useState({ width: 0, height: 0 });

    // Derive the current target based on index
    const currentFeedback = data.feedback_points?.[currentFeedbackIndex];

    // The Sync Engine
    const handlePlaybackStatusUpdate = async (status: AVPlaybackStatus) => {
        if (!status.isLoaded) return;

        // 1. Trigger the specific Coaching Moment
        if (
            currentFeedback &&
            !isPausedForFeedback &&
            !hasTriggeredCurrentFeedback.current &&
            status.positionMillis >= currentFeedback.mistake_timestamp_ms
        ) {

            // Immediately lock to prevent the next 50ms tick from firing this block again
            hasTriggeredCurrentFeedback.current = true;

            if (videoRef.current) {
                await videoRef.current.pauseAsync();
            }

            setIsPausedForFeedback(true);
            setShowingFrame(false);

            if (currentFeedback.audio_url) {
                try {
                    await Audio.setAudioModeAsync({
                        playsInSilentModeIOS: true,
                        shouldDuckAndroid: false
                    });

                    if (soundRef.current) {
                        await soundRef.current.unloadAsync();
                    }

                    // Support both base64-encoded audio from the backend
                    // AND plain URLs used in mock data
                    let audioSource: { uri: string };
                    if (currentFeedback.audio_url.startsWith('data:audio')) {
                        // Backend sends base64 — write raw bytes to a temp file
                        // using the legacy FileSystem API (EncodingType.Base64 = correct binary write)
                        const base64Data = currentFeedback.audio_url.split(',')[1];
                        const tempPath = `${FileSystem.cacheDirectory}ai_voice_${Date.now()}.wav`;
                        await FileSystem.writeAsStringAsync(tempPath, base64Data, {
                            encoding: FileSystem.EncodingType.Base64,
                        });
                        audioSource = { uri: tempPath };
                    } else {
                        // Mock data URL — load directly
                        audioSource = { uri: currentFeedback.audio_url };
                    }

                    const { sound } = await Audio.Sound.createAsync(
                        audioSource,
                        { shouldPlay: true }
                    );

                    soundRef.current = sound;
                } catch (e) {
                    console.error("Failed to play AI Voice on trigger:", e);
                }
            }
        }

        // 2. Video finished naturally
        if (status.didJustFinish) {
            setIsVideoFinished(true);
        }
    };

    const handleShowFrame = async () => {
        setShowingFrame(true);

        // Mute the audio instantly the moment they dismiss the text box to look at the frame
        if (soundRef.current) {
            await soundRef.current.stopAsync();
            await soundRef.current.unloadAsync();
            soundRef.current = null;
        }
    };

    const handleContinue = async () => {
        setIsPausedForFeedback(false);
        setShowingFrame(false);

        // Reset the strict lock for the NEXT feedback point in the sequence
        hasTriggeredCurrentFeedback.current = false;

        // Advance to the next coaching point in the array
        setCurrentFeedbackIndex(prev => prev + 1);

        // Resume video playback
        if (videoRef.current) {
            await videoRef.current.playAsync();
        }
    };

    const finishSession = () => {
        navigation.replace('Complete', { data, activityType, goalId });
    };

    return (
        <View style={globalStyles.fullScreen}>

            <View
                style={S.videoContainer}
                onLayout={(event) => {
                    const { width, height } = event.nativeEvent.layout;
                    setVideoLayout({ width, height });
                }}
            >
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
                                <Text style={S.coachingHeader}>Coach Says</Text>
                                <Text style={S.coachingText}>{currentFeedback.coaching_script}</Text>
                                <Text style={S.timestampText}>Paused at {(currentFeedback.mistake_timestamp_ms / 1000).toFixed(1)}s</Text>

                                <TouchableOpacity
                                    style={S.showFrameBtn}
                                    onPress={handleShowFrame}
                                >
                                    <Text style={S.showFrameBtnText}>Show Corrected Frame  →</Text>
                                </TouchableOpacity>
                            </View>
                        )}

                        {/* STEP 2: The Visible Frame + SVG Overlay + Continue Button */}
                        {showingFrame && (
                            <View style={S.frameOverlayContainer}>

                                <SVGOverlay
                                    data={currentFeedback.visuals}
                                    videoLayout={videoLayout}
                                />

                                <TouchableOpacity
                                    style={S.continueBtn}
                                    onPress={handleContinue}
                                >
                                    <Text style={S.continueBtnText}>
                                        {currentFeedbackIndex < (data.feedback_points?.length || 0) - 1 ? 'Next  →' : 'Continue  →'}
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        )}

                    </View>
                )}
            </View>

            {/* Only show bottom bar when NOT paused for feedback, so it never covers the Next button */}
            {!isPausedForFeedback && (
                <View style={S.bottomControls}>
                    {!isVideoFinished ? (
                        <Text style={S.playbackStatus}>▶ Watching your form…</Text>
                    ) : (
                        <TouchableOpacity
                            style={S.finishBtn}
                            onPress={finishSession}
                        >
                            <Text style={S.finishBtnText}>View Results  →</Text>
                        </TouchableOpacity>
                    )}
                </View>
            )}

        </View>
    );
}

const S = StyleSheet.create({
    videoContainer: {
        flex: 1,                    // Fill all available screen space
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
        padding: 22,
        borderRadius: 20,
        width: '100%',
        borderWidth: 1,
        borderColor: Colors.primaryBorder,
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.25,
        shadowRadius: 20,
        elevation: 12,
    },
    coachingHeader: {
        color: Colors.primary,
        fontSize: 11,
        fontWeight: '800',
        marginBottom: 10,
        textTransform: 'uppercase',
        letterSpacing: 2,
    },
    coachingText: {
        color: Colors.text,
        fontSize: 17,
        lineHeight: 27,
        fontWeight: '500',
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
    showFrameBtn: {
        marginTop: 18,
        backgroundColor: Colors.primary,
        paddingVertical: 14,
        borderRadius: 16,
        alignItems: 'center',
    },
    showFrameBtnText: {
        color: Colors.background,
        fontSize: 15,
        fontWeight: '700',
    },
    continueBtn: {
        backgroundColor: Colors.primary,
        paddingVertical: 16,
        paddingHorizontal: 40,
        borderRadius: 16,
        alignItems: 'center',
        marginBottom: 60,   // clear the home indicator
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.5,
        shadowRadius: 12,
        elevation: 10,
    },
    continueBtnText: {
        color: Colors.background,
        fontSize: 16,
        fontWeight: '800',
    },
    bottomControls: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingVertical: 20,
        paddingBottom: 56,
        alignItems: 'center',
        backgroundColor: 'rgba(10,10,15,0.65)',
    },
    playbackStatus: {
        color: Colors.textSecondary,
        fontSize: 14,
        fontWeight: '600',
        backgroundColor: 'rgba(255,255,255,0.08)',
        paddingHorizontal: 16,
        paddingVertical: 7,
        borderRadius: 20,
        overflow: 'hidden',
    },
    finishBtn: {
        backgroundColor: Colors.primary,
        paddingVertical: 15,
        paddingHorizontal: 48,
        borderRadius: 16,
        alignItems: 'center',
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.4,
        shadowRadius: 14,
        elevation: 10,
    },
    finishBtnText: {
        color: Colors.background,
        fontSize: 17,
        fontWeight: '800',
        letterSpacing: 0.2,
    },
});
