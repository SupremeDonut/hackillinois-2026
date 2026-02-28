import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Recording'>;

export default function RecordingScreen() {
    const navigation = useNavigation<NavigationProp>();

    const handleSimulateRecording = () => {
        // Stage 1 mock
        navigation.replace('Analyzing', {
            videoUri: 'dummy_video_uri.mp4',
            activityType: 'basketball_shot',
            description: 'Working on my jump shot form',
        });
    };

    return (
        <View style={globalStyles.fullScreen}>
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>Recording Screen (Stage 1 Mock)</Text>
                <Text style={globalStyles.subHeading}>Camera will go here in Stage 2.</Text>

                <TouchableOpacity
                    style={globalStyles.primaryButton}
                    onPress={handleSimulateRecording}
                >
                    <Text style={globalStyles.buttonText}>Simulate 5s Recording</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}
