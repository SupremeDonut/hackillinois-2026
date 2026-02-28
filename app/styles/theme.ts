import { StyleSheet } from 'react-native';

export const Colors = {
    background: '#121212', // Pure dark mode
    surface: '#1E1E1E',
    primary: '#4CAF50', // Encouraging green
    text: '#FFFFFF',
    textSecondary: '#AAAAAA',
    error: '#FF5252',
    vectorCurrent: '#FF3B30', // SVG Red
    vectorTarget: '#34C759',  // SVG Green
};

export const globalStyles = StyleSheet.create({
    fullScreen: {
        flex: 1,
        backgroundColor: Colors.background
    },
    centerContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 16,
    },
    heading: {
        fontSize: 24,
        fontWeight: 'bold',
        color: Colors.text,
        marginBottom: 20,
        textAlign: 'center',
    },
    subHeading: {
        fontSize: 16,
        color: Colors.textSecondary,
        marginBottom: 30,
        textAlign: 'center',
    },
    primaryButton: {
        backgroundColor: Colors.primary,
        paddingVertical: 16,
        paddingHorizontal: 32,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 20,
        width: '100%',
    },
    buttonText: {
        color: Colors.text,
        fontSize: 18,
        fontWeight: 'bold',
    },
});
