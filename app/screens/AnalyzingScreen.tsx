import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, AnalysisResponse } from '../types';
import { globalStyles } from '../styles/theme';

// Import our Stage 3 Mock JSON to pass to Playback
import mockResponseData from '../data/mock_response.json';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Analyzing'>;
type AnalyzingRouteProp = RouteProp<RootStackParamList, 'Analyzing'>;

export default function AnalyzingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<AnalyzingRouteProp>();
    const { videoUri, activityType } = route.params;

    useEffect(() => {
        // Stage 3 & Stage 1 mock: Simulate API delay then move to Playback
        // We cast mockResponseData as any first because JSON imports don't narrowly match the string enums exactly
        const typedMockData = mockResponseData as any as AnalysisResponse;

        const timer = setTimeout(() => {
            navigation.replace('Playback', {
                videoUri,
                data: typedMockData,
                activityType,
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
