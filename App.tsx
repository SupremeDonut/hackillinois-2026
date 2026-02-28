import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'react-native';

import HomeScreen from './app/screens/HomeScreen';
import RecordingScreen from './app/screens/RecordingScreen';
import AnalyzingScreen from './app/screens/AnalyzingScreen';
import PlaybackScreen from './app/screens/PlaybackScreen';

const Stack = createNativeStackNavigator();

export default function App() {
    return (
        <NavigationContainer>
            <StatusBar barStyle="light-content" />
            <Stack.Navigator
                initialRouteName="Home"
                screenOptions={{
                    headerShown: false,
                    contentStyle: { backgroundColor: '#09090b' }, // dark zinc background
                    animation: 'fade', // Smooth transitions
                }}
            >
                <Stack.Screen name="Home" component={HomeScreen} />
                <Stack.Screen name="Recording" component={RecordingScreen} />
                <Stack.Screen name="Analyzing" component={AnalyzingScreen} />
                <Stack.Screen name="Playback" component={PlaybackScreen} />
            </Stack.Navigator>
        </NavigationContainer>
    );
}
