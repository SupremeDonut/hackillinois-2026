import React, { useEffect, useState } from 'react';
import { NavigationContainer, DarkTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { RootStackParamList } from './app/types';
import { getAccount } from './app/services/accountStore';

// Screens
import OnboardingScreen from './app/screens/OnboardingScreen';
import HomeScreen from './app/screens/HomeScreen';
import RecordingScreen from './app/screens/RecordingScreen';
import AnalyzingScreen from './app/screens/AnalyzingScreen';
import PlaybackScreen from './app/screens/PlaybackScreen';
import CompleteScreen from './app/screens/CompleteScreen';
import GoalDetailScreen from './app/screens/GoalDetailScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
    const [initialRoute, setInitialRoute] = useState<keyof RootStackParamList | null>(null);

    useEffect(() => {
        getAccount().then((account) => {
            setInitialRoute(account ? 'Home' : 'Onboarding');
        });
    }, []);

    if (!initialRoute) return null;

    return (
        <NavigationContainer theme={DarkTheme}>
            <Stack.Navigator
                initialRouteName={initialRoute}
                screenOptions={{
                    headerShown: false,
                    animation: 'slide_from_right'
                }}
            >
                <Stack.Screen name="Onboarding" component={OnboardingScreen} />
                <Stack.Screen name="Home" component={HomeScreen} />
                <Stack.Screen name="Recording" component={RecordingScreen} />
                <Stack.Screen name="Analyzing" component={AnalyzingScreen} />
                <Stack.Screen name="Playback" component={PlaybackScreen} />
                <Stack.Screen name="Complete" component={CompleteScreen} />
                <Stack.Screen name="GoalDetail" component={GoalDetailScreen} />
            </Stack.Navigator>
        </NavigationContainer>
    );
}
