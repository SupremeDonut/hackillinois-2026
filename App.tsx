import React from 'react';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RootStackParamList } from './app/types';

// Screens
import HomeScreen from './app/screens/HomeScreen';
import RecordingScreen from './app/screens/RecordingScreen';
import AnalyzingScreen from './app/screens/AnalyzingScreen';
import PlaybackScreen from './app/screens/PlaybackScreen';
import CompleteScreen from './app/screens/CompleteScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
    return (
        <NavigationContainer theme={DarkTheme}>
            <Stack.Navigator
                initialRouteName="Home"
                screenOptions={{
                    headerShown: false,
                    animation: 'slide_from_right'
                }}
            >
                <Stack.Screen name="Home" component={HomeScreen} />
                <Stack.Screen name="Recording" component={RecordingScreen} />
                <Stack.Screen name="Analyzing" component={AnalyzingScreen} />
                <Stack.Screen name="Playback" component={PlaybackScreen} />
                <Stack.Screen name="Complete" component={CompleteScreen} />
            </Stack.Navigator>
        </NavigationContainer>
    );
}
