import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView, Dimensions } from 'react-native';
import { Video, ResizeMode, Audio } from 'expo-av';
import SVGOverlay from '../components/SVGOverlay';

const screenWidth = Dimensions.get('window').width;
const videoHeight = screenWidth * (16 / 9);

export default function PlaybackScreen({ route, navigation }: any) {
    const { analysis, originalVideoUri } = route.params;

    const videoRef = useRef<Video>(null);
    const audioRef = useRef<Audio.Sound | null>(null);

    const [isPaused, setIsPaused] = useState(false);
    const [showOverlay, setShowOverlay] = useState(false);
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        // Setup Audio Configuration so ElevenLabs can play over muted video
        const setupAudio = async () => {
            try {
                await Audio.setAudioModeAsync({
                    playsInSilentModeIOS: true,
                    staysActiveInBackground: true,
                    shouldDuckAndroid: false,
                });
                if (analysis.audio_url) {
                    const { sound } = await Audio.Sound.createAsync(
                        { uri: analysis.audio_url }
                    );
                    audioRef.current = sound;
                    // Play the audio feedback once the UI mounts
                    await sound.playAsync();
                }
            } catch (error) {
                console.log("Audio play error", error);
            }
        };

        setupAudio();

        return () => {
            if (audioRef.current) {
                audioRef.current.unloadAsync();
            }
        };
    }, []);

    const handlePlaybackStatusUpdate = (status: any) => {
        if (!status.isLoaded) return;

        // Has the video finished?
        if (status.didJustFinish) {
            setIsComplete(true);
            return;
        }

        // Has the video hit the exact mistake frame for the first time?
        if (
            status.positionMillis >= analysis.mistake_timestamp_ms &&
            !isPaused &&
            !showOverlay &&
            !isComplete
        ) {
            if (videoRef.current) {
                videoRef.current.pauseAsync();
            }
            setIsPaused(true);
            setShowOverlay(true);
        }
    };

    const handleContinue = async () => {
        setShowOverlay(false);
        setIsPaused(false);
        if (videoRef.current) {
            await videoRef.current.playAsync();
        }
    };

    if (isComplete) {
        return (
            <SafeAreaView style={styles.container}>
                <View style={styles.completeContent}>
                    <Text style={styles.completeTitle}>Session Complete</Text>

                    <View style={styles.scoreCard}>
                        <Text style={styles.scoreText}>{analysis.progress_score}</Text>
                        <Text style={styles.scoreSubtext}>Form Score</Text>
                        <Text style={styles.deltaText}>
                            {analysis.improvement_delta > 0 ? `+${analysis.improvement_delta} from last time` : `${analysis.improvement_delta} from last time`}
                        </Text>
                    </View>

                    <View style={styles.feedbackCard}>
                        <Text style={styles.positiveNote}>{analysis.positive_note}</Text>
                        <Text style={styles.scriptNote}>{analysis.coaching_script}</Text>
                    </View>

                    <View style={styles.actionButtons}>
                        <TouchableOpacity style={styles.retryBtn} onPress={() => navigation.replace('Recording')}>
                            <Text style={styles.retryBtnText}>Try Again</Text>
                        </TouchableOpacity>
                        <TouchableOpacity style={styles.doneBtn} onPress={() => navigation.replace('Home')}>
                            <Text style={styles.doneBtnText}>Done</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </SafeAreaView>
        );
    }

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <Text style={styles.title}>Analysis</Text>
            </View>

            <View style={styles.videoContainer}>
                <Video
                    ref={videoRef}
                    style={styles.video}
                    source={{ uri: originalVideoUri || 'http://d23dyxeqlo5psv.cloudfront.net/big_buck_bunny.mp4' }}
                    resizeMode={ResizeMode.COVER}
                    shouldPlay
                    isMuted={true} // Crucial: Mute video so TTS audio overrides
                    progressUpdateIntervalMillis={50} // Crucial for millisecond-precision pausing
                    onPlaybackStatusUpdate={handlePlaybackStatusUpdate}
                />

                {showOverlay && (
                    <View style={styles.overlayContainer}>
                        <SVGOverlay data={analysis.visuals} />

                        <View style={styles.pauseMenu}>
                            <Text style={styles.pauseNote}>Pay attention to the colored overlay</Text>
                            <TouchableOpacity style={styles.continueBtn} onPress={handleContinue}>
                                <Text style={styles.continueBtnText}>Got it, Continue</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                )}
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#09090b' },
    header: { padding: 20, alignItems: 'center' },
    title: { fontSize: 20, fontWeight: '700', color: '#fff' },

    videoContainer: {
        width: screenWidth,
        height: videoHeight,
        backgroundColor: '#18181b',
        position: 'relative',
        overflow: 'hidden'
    },
    video: { flex: 1 },
    overlayContainer: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.2)'
    },
    pauseMenu: {
        position: 'absolute',
        bottom: 50,
        left: 20,
        right: 20,
        alignItems: 'center',
        padding: 20,
        backgroundColor: 'rgba(24,24,27,0.85)',
        borderRadius: 16,
        borderWidth: 1,
        borderColor: '#3f3f46'
    },
    pauseNote: { color: '#e4e4e7', fontSize: 14, marginBottom: 16, textAlign: 'center' },
    continueBtn: { backgroundColor: '#3b82f6', paddingVertical: 14, paddingHorizontal: 32, borderRadius: 100, width: '100%', alignItems: 'center' },
    continueBtnText: { color: '#ffffff', fontSize: 16, fontWeight: '700' },

    // Complete Screen Styles
    completeContent: { flex: 1, padding: 24, justifyContent: 'center' },
    completeTitle: { fontSize: 32, fontWeight: '800', color: '#fff', textAlign: 'center', marginBottom: 40 },
    scoreCard: { alignItems: 'center', marginBottom: 40 },
    scoreText: { fontSize: 80, fontWeight: '900', color: '#10b981', letterSpacing: -2 },
    scoreSubtext: { fontSize: 18, color: '#a1a1aa', fontWeight: '600' },
    deltaText: { fontSize: 14, color: '#34d399', marginTop: 8 },

    feedbackCard: { backgroundColor: '#18181b', padding: 20, borderRadius: 16, borderWidth: 1, borderColor: '#27272a', marginBottom: 40, gap: 12 },
    positiveNote: { color: '#10b981', fontSize: 16, fontWeight: '700' },
    scriptNote: { color: '#e4e4e7', fontSize: 16, lineHeight: 24 },

    actionButtons: { gap: 16 },
    retryBtn: { backgroundColor: '#3b82f6', paddingVertical: 18, borderRadius: 100, alignItems: 'center' },
    retryBtnText: { color: '#ffffff', fontSize: 18, fontWeight: '700' },
    doneBtn: { backgroundColor: 'transparent', paddingVertical: 18, borderRadius: 100, alignItems: 'center', borderWidth: 2, borderColor: '#3f3f46' },
    doneBtnText: { color: '#a1a1aa', fontSize: 18, fontWeight: '700' }
});
