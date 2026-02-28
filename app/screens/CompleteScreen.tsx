import React from 'react';
import { View, Text, TouchableOpacity } from 'react-native';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { RootStackParamList } from '../types';
import { globalStyles } from '../styles/theme';

type NavigationProp = NativeStackNavigationProp<RootStackParamList, 'Complete'>;
type CompleteRouteProp = RouteProp<RootStackParamList, 'Complete'>;

export default function CompleteScreen() {
    const navigation = useNavigation<NavigationProp>();
    const route = useRoute<CompleteRouteProp>();
    const { data } = route.params;

    return (
        <View style={globalStyles.fullScreen}>
            <View style={globalStyles.centerContent}>
                <Text style={globalStyles.heading}>Session Complete</Text>
                <Text style={globalStyles.subHeading}>Progress Score: {data.analysis.progress_score}</Text>
                <Text style={globalStyles.subHeading}>{data.analysis.positive_note}</Text>

                <TouchableOpacity
                    style={globalStyles.primaryButton}
                    onPress={() => navigation.navigate('Recording')}
                >
                    <Text style={globalStyles.buttonText}>Try Again</Text>
                </TouchableOpacity>

                <TouchableOpacity
                    style={[globalStyles.primaryButton, { backgroundColor: '#333', marginTop: 10 }]}
                    onPress={() => navigation.navigate('Home')}
                >
                    <Text style={globalStyles.buttonText}>Done</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
}
