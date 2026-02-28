import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, AnalysisResponse } from '../types';
import { globalStyles } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Analyzing'>;
type AnalyzingRouteProp = RouteProp<RootStackParamList, 'Analyzing'>;

// Dummy data for Stage 1 flow
const mockAnalysisData: AnalysisResponse = {
    status: 'success',
    analysis: {
        mistake_timestamp_ms: 2500,
        coaching_script: 'Great shot! But try to keep your elbow tucked in.',
        positive_note: 'Excellent follow through.',
        progress_score: 85,
    },
    visuals: {
        focus_point: { x: 0.5, y: 0.5 },
        overlay_type: 'ANGLE_CORRECTION',
        vectors: [],
    },
    audio_url: 'dummy_url'
};

export default function AnalyzingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<AnalyzingRouteProp>();
    const { videoUri } = route.params;

    useEffect(() => {
        // Stage 1 mock: Simulate API delay then move to Playback
        const timer = setTimeout(() => {
            navigation.replace('Playback', {
                videoUri,
                data: mockAnalysisData,
            });
        }, 2000);

        return () => clearTimeout(timer);
    }, [navigation, videoUri]);

    return (
        <View style={globalStyles.fullScreen}>
            <View style={globalStyles.centerContent}>
                <ActivityIndicator size="large" color="#4CAF50" />
                <Text style={[globalStyles.heading, { marginTop: 20 }]}>Analyzing your form...</Text>
                <Text style={globalStyles.subHeading}>Hold tight! Our AI is reviewing your video.</Text>
            </View>
        </View>
    );
}
