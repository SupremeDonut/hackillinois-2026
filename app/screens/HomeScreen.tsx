import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, SafeAreaView } from 'react-native';

export default function HomeScreen({ navigation }: any) {
    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.content}>
                <View style={styles.header}>
                    <Text style={styles.title}>Motion<Text style={styles.titleAccent}>Coach</Text></Text>
                    <Text style={styles.subtitle}>AI coaching for your physical hobbies</Text>
                </View>

                <View style={styles.features}>
                    <FeatureItem icon="🎥" title="Record 5s Video" desc="Show us your swing, shot, or move." />
                    <FeatureItem icon="🧠" title="AI Analysis" desc="Gemini identifies your mistake." />
                    <FeatureItem icon="✍️" title="Visual Feedback" desc="On-screen drawings + voice coaching." />
                </View>

                <TouchableOpacity
                    style={styles.primaryButton}
                    activeOpacity={0.8}
                    onPress={() => navigation.navigate('Recording')}
                >
                    <Text style={styles.primaryButtonText}>Start New Session</Text>
                </TouchableOpacity>
            </View>
        </SafeAreaView>
    );
}

const FeatureItem = ({ icon, title, desc }: { icon: string, title: string, desc: string }) => (
    <View style={styles.featureItem}>
        <Text style={styles.featureIcon}>{icon}</Text>
        <View style={styles.featureText}>
            <Text style={styles.featureTitle}>{title}</Text>
            <Text style={styles.featureDesc}>{desc}</Text>
        </View>
    </View>
);

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#09090b', // Zinc 950
    },
    content: {
        flex: 1,
        padding: 24,
        justifyContent: 'space-between',
    },
    header: {
        marginTop: 60,
    },
    title: {
        fontSize: 42,
        fontWeight: '900',
        color: '#ffffff',
        letterSpacing: -1,
    },
    titleAccent: {
        color: '#3b82f6', // Blue 500
    },
    subtitle: {
        fontSize: 18,
        color: '#a1a1aa', // Zinc 400
        marginTop: 12,
        fontWeight: '500',
    },
    features: {
        marginVertical: 40,
        gap: 24,
    },
    featureItem: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#18181b', // Zinc 900
        padding: 20,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: '#27272a', // Zinc 800
    },
    featureIcon: {
        fontSize: 32,
        marginRight: 16,
    },
    featureText: {
        flex: 1,
    },
    featureTitle: {
        color: '#ffffff',
        fontSize: 18,
        fontWeight: '700',
        marginBottom: 4,
    },
    featureDesc: {
        color: '#a1a1aa',
        fontSize: 14,
        lineHeight: 20,
    },
    primaryButton: {
        backgroundColor: '#3b82f6',
        paddingVertical: 18,
        borderRadius: 100,
        alignItems: 'center',
        shadowColor: '#3b82f6',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
        marginBottom: 20,
    },
    primaryButtonText: {
        color: '#ffffff',
        fontSize: 18,
        fontWeight: '700',
    }
});
