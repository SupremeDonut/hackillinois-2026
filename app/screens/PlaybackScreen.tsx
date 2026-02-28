import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Playback'>;
type PlaybackRouteProp = RouteProp<RootStackParamList, 'Playback'>;

export default function PlaybackScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<PlaybackRouteProp>();
    const { videoUri, data } = route.params;

    const handleSimulateFinish = () => {
        // Stage 1 mock
        navigation.replace('Complete', { data });
    };

    return (
        <View style={globalStyles.fullScreen}>
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>Playback Screen (Stage 1 Mock)</Text>
                <Text style={globalStyles.subHeading}>Video payload: {videoUri}</Text>
                <Text style={globalStyles.subHeading}>Feedback: {data.analysis.coaching_script}</Text>

                <TouchableOpacity
                    style={globalStyles.primaryButton}
                    onPress={handleSimulateFinish}
                >
                    <Text style={globalStyles.buttonText}>Simulate Video Finish</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}
