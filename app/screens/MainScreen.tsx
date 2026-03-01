import React, { useState } from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import Svg, { Path, Circle, Rect } from 'react-native-svg';
import { Colors, Radius } from '../styles/theme';
import HomeScreen from './HomeScreen';
import GoalsTab from './GoalsTab';
import HistoryTab from './HistoryTab';

type Tab = 'record' | 'goals' | 'history';

// Simple SVG icons for each tab
function RecordIcon({ active }: { active: boolean }) {
    const c = active ? Colors.primary : Colors.textMuted;
    return (
        <Svg width={22} height={22} viewBox="0 0 22 22">
            <Circle cx="11" cy="11" r="7" stroke={c} strokeWidth="1.8" fill="none" />
            <Circle cx="11" cy="11" r="3.5" fill={c} />
        </Svg>
    );
}

function GoalsIcon({ active }: { active: boolean }) {
    const c = active ? Colors.primary : Colors.textMuted;
    return (
        <Svg width={22} height={22} viewBox="0 0 22 22">
            <Path
                d="M11 2 L13.5 8H20L14.5 12L17 18L11 14L5 18L7.5 12L2 8H8.5Z"
                stroke={c} strokeWidth="1.6" strokeLinejoin="round" fill="none"
            />
        </Svg>
    );
}

function HistoryIcon({ active }: { active: boolean }) {
    const c = active ? Colors.primary : Colors.textMuted;
    return (
        <Svg width={22} height={22} viewBox="0 0 22 22">
            <Rect x="3" y="5" width="16" height="2.2" rx="1.1" fill={c} />
            <Rect x="3" y="9.9" width="12" height="2.2" rx="1.1" fill={c} />
            <Rect x="3" y="14.8" width="8" height="2.2" rx="1.1" fill={c} />
        </Svg>
    );
}

const TABS: { id: Tab; label: string }[] = [
    { id: 'record', label: 'Record' },
    { id: 'goals', label: 'Goals' },
    { id: 'history', label: 'History' },
];

export default function MainScreen() {
    const [activeTab, setActiveTab] = useState<Tab>('record');
    const insets = useSafeAreaInsets();

    return (
        <View style={S.container}>
            {/* Tab content */}
            <View style={S.content}>
                {activeTab === 'record' && <HomeScreen />}
                {activeTab === 'goals' && <GoalsTab />}
                {activeTab === 'history' && <HistoryTab />}
            </View>

            {/* Bottom tab bar */}
            <View style={[S.tabBar, { paddingBottom: Math.max(insets.bottom, 10) }]}>
                {TABS.map(tab => {
                    const isActive = activeTab === tab.id;
                    return (
                        <TouchableOpacity
                            key={tab.id}
                            style={S.tabItem}
                            onPress={() => setActiveTab(tab.id)}
                            activeOpacity={0.7}
                        >
                            {tab.id === 'record' && <RecordIcon active={isActive} />}
                            {tab.id === 'goals' && <GoalsIcon active={isActive} />}
                            {tab.id === 'history' && <HistoryIcon active={isActive} />}
                            <Text style={[S.tabLabel, isActive && S.tabLabelActive]}>
                                {tab.label}
                            </Text>
                            {isActive && <View style={S.activeDot} />}
                        </TouchableOpacity>
                    );
                })}
            </View>
        </View>
    );
}

const S = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    content: {
        flex: 1,
    },
    tabBar: {
        flexDirection: 'row',
        backgroundColor: Colors.surface,
        borderTopWidth: 1,
        borderTopColor: Colors.glassBorder,
        paddingTop: 10,
        ...Platform.select({
            ios: {
                shadowColor: '#000',
                shadowOffset: { width: 0, height: -4 },
                shadowOpacity: 0.15,
                shadowRadius: 8,
            },
            android: {
                elevation: 12,
            },
        }),
    },
    tabItem: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 3,
        position: 'relative',
    },
    tabLabel: {
        fontSize: 10,
        fontWeight: '600',
        color: Colors.textMuted,
        letterSpacing: 0.5,
        textTransform: 'uppercase',
    },
    tabLabelActive: {
        color: Colors.primary,
    },
    activeDot: {
        position: 'absolute',
        bottom: -8,
        width: 4,
        height: 4,
        borderRadius: 2,
        backgroundColor: Colors.primary,
    },
});
