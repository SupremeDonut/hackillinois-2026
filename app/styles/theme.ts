import { StyleSheet, Platform } from 'react-native';

// ─── Design Tokens ───────────────────────────────────────────────────────────

export const Colors = {
    // Core palette
    background: '#16161F',       // Dark navy-charcoal (lifted from near-black)
    backgroundAlt: '#1E1E2C',    // Slightly lighter for layering
    surface: '#26263A',          // Card / input surface
    surfaceHighlight: '#30304A', // Hover / pressed state

    // Accent — electric teal-green
    primary: '#00E5A0',
    primaryDim: 'rgba(0, 229, 160, 0.18)',
    primaryBorder: 'rgba(0, 229, 160, 0.40)',

    // Status colours
    error: '#F07070',             // Soft rose
    errorDim: 'rgba(240, 112, 112, 0.18)',
    success: '#90EDB0',           // Light mint (delta positive)

    // Text
    text: '#F4F4FC',
    textSecondary: '#9898B8',
    textMuted: '#66667E',

    // SVG overlay colours (unchanged — backend uses these)
    vectorCurrent: '#FF3B30',
    vectorTarget: '#34C759',

    // Glassmorphism border
    glassBorder: 'rgba(255,255,255,0.11)',
    glassBg: 'rgba(255,255,255,0.06)',
};

export const Spacing = {
    xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48,
};

export const Radius = {
    sm: 8, md: 12, lg: 20, xl: 28, full: 999,
};

export const Shadow = {
    card: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.35,
        shadowRadius: 12,
        elevation: 8,
    },
    glow: {
        shadowColor: Colors.primary,
        shadowOffset: { width: 0, height: 0 },
        shadowOpacity: 0.4,
        shadowRadius: 16,
        elevation: 10,
    },
};

// ─── Global Styles ────────────────────────────────────────────────────────────

export const globalStyles = StyleSheet.create({
    // Screens
    fullScreen: {
        flex: 1,
        backgroundColor: Colors.background,
    },
    centerContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: Spacing.lg,
        backgroundColor: Colors.background,
    },

    // Typography
    heading: {
        fontSize: 28,
        fontWeight: '800',
        color: Colors.text,
        marginBottom: Spacing.sm,
        textAlign: 'center',
        letterSpacing: -0.5,
    },
    headingLarge: {
        fontSize: 36,
        fontWeight: '800',
        color: Colors.text,
        marginBottom: Spacing.md,
        letterSpacing: -1,
    },
    subHeading: {
        fontSize: 15,
        color: Colors.textSecondary,
        marginBottom: Spacing.lg,
        textAlign: 'center',
        lineHeight: 22,
    },
    label: {
        fontSize: 11,
        fontWeight: '700',
        color: Colors.textSecondary,
        textTransform: 'uppercase',
        letterSpacing: 1.2,
        marginBottom: Spacing.sm,
    },

    // Buttons
    primaryButton: {
        backgroundColor: Colors.primary,
        paddingVertical: 16,
        paddingHorizontal: Spacing.xl,
        borderRadius: Radius.lg,
        alignItems: 'center',
        marginTop: Spacing.md,
        width: '100%',
        ...Shadow.glow,
    },
    secondaryButton: {
        backgroundColor: Colors.surface,
        paddingVertical: 16,
        paddingHorizontal: Spacing.xl,
        borderRadius: Radius.lg,
        alignItems: 'center',
        marginTop: Spacing.sm,
        width: '100%',
        borderWidth: 1,
        borderColor: Colors.glassBorder,
    },
    buttonText: {
        color: Colors.background,  // Dark text on primary (contrast on teal)
        fontSize: 16,
        fontWeight: '700',
        letterSpacing: 0.3,
    },
    buttonTextSecondary: {
        color: Colors.text,
        fontSize: 16,
        fontWeight: '600',
    },

    // Cards
    card: {
        backgroundColor: Colors.surface,
        borderRadius: Radius.lg,
        padding: Spacing.lg,
        borderWidth: 1,
        borderColor: Colors.glassBorder,
        ...Shadow.card,
    },

    // Divider
    divider: {
        height: 1,
        backgroundColor: Colors.glassBorder,
        marginVertical: Spacing.md,
    },
});
