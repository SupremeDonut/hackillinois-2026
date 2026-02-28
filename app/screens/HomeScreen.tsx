import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

export default function HomeScreen() {
    const navigation = useNavigation<NavigationProp>();

    return (
        <View style={globalStyles.fullScreen}>
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>MotionCoach AI</Text>
                <Text style={globalStyles.subHeading}>AI coaching for your physical hobbies.</Text>

                <TouchableOpacity
                    style={globalStyles.primaryButton}
                    onPress={() => navigation.navigate('Recording')}
                >
                    <Text style={globalStyles.buttonText}>Start Recording</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}
