import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, TextInput, KeyboardAvoidingView, Platform } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList, ActivityType } from '../types';
import { globalStyles, Colors } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

export default function HomeScreen() {
    const navigation = useNavigation<NavigationProp>();
    const [activityType, setActivityType] = useState('');

    const isReady = activityType.trim().length > 0;

    return (
        <KeyboardAvoidingView
            behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
            style={globalStyles.fullScreen}
        >
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>MotionCoach AI</Text>
                <Text style={globalStyles.subHeading}>AI coaching for your physical hobbies.</Text>

                <View style={S.formContainer}>
                    <Text style={S.inputLabel}>What are you practicing?</Text>

                    <TextInput
                        style={S.activityInput}
                        value={activityType}
                        onChangeText={setActivityType}
                        placeholder="type here or select below"
                        placeholderTextColor={Colors.textSecondary}
                    />

                    <View style={S.suggestionsContainer}>
                        {['Basketball Shot', 'Badminton Smash', 'Guitar G Chord'].map((item) => (
                            <TouchableOpacity
                                key={item}
                                style={[S.suggestionChip, activityType === item && S.suggestionChipActive]}
                                onPress={() => setActivityType(item)}
                            >
                                <Text style={[S.suggestionText, activityType === item && S.suggestionTextActive]}>
                                    {item}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </View>
                </View>

                <TouchableOpacity
                    style={[globalStyles.primaryButton, !isReady && S.disabledButton]}
                    onPress={() => isReady && navigation.navigate('Recording', { activityType })}
                    disabled={!isReady}
                >
                    <Text style={globalStyles.buttonText}>Start Recording</Text>
                </TouchableOpacity>
            </View>
        </KeyboardAvoidingView>
    );
}

const S = StyleSheet.create({
    formContainer: {
        width: '100%',
        marginVertical: 40,
    },
    inputLabel: {
        color: Colors.textSecondary,
        fontSize: 14,
        alignSelf: 'flex-start',
        marginBottom: 8,
        fontWeight: 'bold',
        textTransform: 'uppercase'
    },
    activityInput: {
        width: '100%',
        backgroundColor: Colors.surface,
        color: Colors.text,
        padding: 16,
        borderRadius: 12,
        fontSize: 16,
        borderWidth: 1,
        borderColor: '#333',
        marginBottom: 16
    },
    suggestionsContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
        width: '100%',
    },
    suggestionChip: {
        backgroundColor: Colors.surface,
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: '#333'
    },
    suggestionChipActive: {
        backgroundColor: 'rgba(76, 175, 80, 0.2)', // Light primary
        borderColor: Colors.primary
    },
    suggestionText: {
        color: Colors.textSecondary,
        fontSize: 14
    },
    suggestionTextActive: {
        color: Colors.primary,
        fontWeight: 'bold'
    },
    disabledButton: {
        backgroundColor: '#333',
        opacity: 0.5
    }
});
