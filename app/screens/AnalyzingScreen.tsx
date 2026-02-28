import React, { useEffect } from 'react';
import { View, Text, ActivityIndicator } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, AnalysisResponse } from '../types';
import { globalStyles } from '../styles/theme';
import { uploadVideo } from '../services/api';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Analyzing'>;
type AnalyzingRouteProp = RouteProp<RootStackParamList, 'Analyzing'>;

export default function AnalyzingScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<AnalyzingRouteProp>();
    const { videoUri, activityType, description, previousData } = route.params;

    useEffect(() => {
        let cancelled = false;

        const runAnalysis = async () => {
            // Calls the API service — returns real data if MODAL_API_URL is set,
            // or instantly falls back to mock_response.json if it's null or fails.
            const data = await uploadVideo({
                videoUri,
                activityType,
                description,
                previousData,
            });

            if (!cancelled) {
                navigation.replace('Playback', {
                    videoUri,
                    data: data as AnalysisResponse,
                    activityType,
                });
            }
        };

        runAnalysis();

        return () => { cancelled = true; };
    }, []);

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
