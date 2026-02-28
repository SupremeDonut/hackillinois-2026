/**
 * ==============================================================================
 * 🚀 [DEV 1/ALL] MAIN APP ENTRYPOINT (app/App.tsx)
 * ==============================================================================
 * Purpose:
 *   This is the root component of the React Native Expo application. 
 *   It sets up the React Navigation container and defines the screen routes
 *   (e.g., Recording, Playback, Home, Goals) that the user navigates between.
 *
 * Dev Instructions:
 *   - Dev 1: Register new screens here as you build them.
 *   - Ensure `initialRouteName` is correct for testing (e.g., set it to 
 *     "Recording" during development to skip directly to capturing video).
 * ==============================================================================
 */
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

import RecordingScreen from './screens/RecordingScreen';
import PlaybackScreen from './screens/PlaybackScreen';
// Dev 1: Import other screens here (e.g., HomeScreen, GoalsScreen)

const Stack = createNativeStackNavigator();

export default function App() {
    return (
        <NavigationContainer>
            <Stack.Navigator initialRouteName="Recording">
                <Stack.Screen name="Recording" component={RecordingScreen} options={{ headerShown: false }} />
                <Stack.Screen name="Playback" component={PlaybackScreen} options={{ headerShown: false }} />
                {/* Dev 1: Add new screens below */}
            </Stack.Navigator>
        </NavigationContainer>
    );
}
