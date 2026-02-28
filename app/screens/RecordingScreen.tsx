/**
 * ==============================================================================
 * 🎥 [DEV 1/DEV 2] CAMERA AND UPLOAD SCREEN (app/screens/RecordingScreen.tsx)
 * ==============================================================================
 * Purpose:
 *   Handles the `expo-camera` capturing logic. The user has 5 seconds maximum
 *   to record their physical action (e.g., throwing a baseball).
 *
 * Responsibilities:
 *   1. Ask for camera/mic permissions.
 *   2. Render the <CameraView> properly mapped to the screen.
 *   3. Enforce the 5-second countdown timer to keep the app snappy.
 *   4. Feed the resulting file URI to `services/api.ts` to push to Modal.
 *
 * The Aspect Ratio Trap (CRITICAL):
 *   - Most camera sensors do not match a phone screen. E.g., a phone screen
 *     is 19.5:9, but the camera captures natively at 4:3 or 16:9.
 *   - If you let React Native dynamically stretch (`resizeMode="cover"`) the 
 *     video, the X/Y coordinate math that Dev 3 is building will BREAK.
 *   - Dev 1 MUST lock the camera's aspect ratio (e.g., explicitly 16:9) 
 *     so the captured video's bounding-box is exactly known.
 * ==============================================================================
 */
import React, { useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
// import { CameraView } from 'expo-camera';
import { uploadVideo } from '../services/api';

export default function RecordingScreen({ navigation }: any) {
    const [isRecording, setIsRecording] = useState(false);

    // Starts the camera and caps the session at 5 seconds
    const startRecording = async () => {
        // Implement 5-second max recording logic here
        console.log("Started recording...");
    };

    // Callback when recording completes. Takes the raw file string.
    const handleUpload = async (fileUri: string) => {
        // 1. Gather context
        const metadata = {
            activity_type: 'tennis_serve',
            user_description: 'fix my serve'
        };

        // 2. Upload file via the 'multipart/form-data' API handler
        // const result = await uploadVideo(fileUri, metadata);

        // 3. Move the user to the Playback engine and pass the Modal response
        // navigation.navigate('Playback', { analysis: result });
    };

    return (
        <View style={styles.container}>
            {/* Dev 1: Insert CameraView here locked to 16:9 */}
            <Text style={styles.text}>Recording Screen</Text>
            <Text style={styles.subtext}>(CameraView goes here)</Text>

            <Button title="Record (5s)" onPress={startRecording} />
            {/* Dev 3: Test SVGOverlay without uploading — navigates to Playback with mock visuals. */}
            <Button
                title="Test overlay (mock data)"
                onPress={() => navigation.navigate('Playback', {})}
            />
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#000',
    },
    text: {
        color: '#fff',
        fontSize: 20,
        marginBottom: 10
    },
    subtext: {
        color: '#aaa',
        marginBottom: 20
    }
});
