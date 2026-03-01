import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = '@morphi_account';

export interface Account {
    displayName: string;
    email: string;
    createdAt: string;
}

export async function getAccount(): Promise<Account | null> {
    try {
        const raw = await AsyncStorage.getItem(KEY);
        return raw ? (JSON.parse(raw) as Account) : null;
    } catch {
        return null;
    }
}

export async function saveAccount(account: Account): Promise<void> {
    await AsyncStorage.setItem(KEY, JSON.stringify(account));
}

export async function clearAccount(): Promise<void> {
    await AsyncStorage.removeItem(KEY);
}
