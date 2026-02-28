import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, SafeAreaView, ActivityIndicator } from 'react-native';
import { uploadVideo } from '../services/api';

const TIPS = [
    "Analyzing your biomechanics...",
    "Sending video to Gemini 3.1 Flash...",
    "Calculating joint angles...",
    "Generating audio feedback..."
];

export default function AnalyzingScreen({ route, navigation }: any) {
    const { fileUri, metadata } = route.params;
    const [tipIndex, setTipIndex] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setTipIndex((prev) => (prev + 1) % TIPS.length);
        }, 2500);
        return () => clearInterval(interval);
    }, []);

    useEffect(() => {
        const processVideo = async () => {
            try {
                // Call the upload function in api.ts
                const analysisResult = await uploadVideo(fileUri, metadata);

                // Immediately jump to playback when done
                navigation.replace('Playback', { analysis: analysisResult, originalVideoUri: fileUri });
            } catch (error) {
                console.error("Analysis Error:", error);
                // Fallback to playback with mock data handled inside api.ts usually,
                // but just in case, we could redirect back to Home.
            }
        };

        processVideo();
    }, [fileUri, metadata]);

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.content}>
                <View style={styles.pulsar}>
                    <ActivityIndicator size="large" color="#3b82f6" />
                </View>

                <Text style={styles.title}>Coach is thinking...</Text>

                <View style={styles.tipContainer}>
                    <Text style={styles.tipText}>{TIPS[tipIndex]}</Text>
                </View>
            </View>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#09090b', // Zinc 950
    },
    content: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    pulsar: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: '#1d4ed833',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 30,
    },
    title: {
        fontSize: 24,
        fontWeight: '800',
        color: '#ffffff',
        marginBottom: 10,
    },
    tipContainer: {
        height: 40,
        justifyContent: 'center',
    },
    tipText: {
        fontSize: 16,
        color: '#a1a1aa', // Zinc 400
        fontWeight: '500',
    },
});
