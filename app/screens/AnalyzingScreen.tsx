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
            // When previousData is present and no real backend is set,
            // use the retry mock so the improvement delta UI can be tested.
            const data = await uploadVideo({
                videoUri,
                activityType,
                description,
                previousData,
                _useMockRetry: !!previousData,  // hint to api.ts to use the retry payload
            } as any);

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
