/**
 * ==============================================================================
 * 🌐 [DEV 1/DEV 2] MODAL API UPLOAD HANDLER (app/services/api.ts)
 * ==============================================================================
 */

export const uploadVideo = async (fileUri: string, metadata: any) => {
    // 1. Instantiate the multipart form object
    const formData = new FormData();

    // 2. Format binary specifically for React Native's FormData polyfill
    formData.append('video_file', {
        uri: fileUri,
        name: `video_${Date.now()}.mp4`,
        type: 'video/mp4',
    } as any);

    // 3. Attach metadata
    formData.append('activity_type', metadata.activity_type);
    formData.append('user_description', metadata.user_description);

    try {
        console.log("Uploading video to Modal...", metadata);
        // FOR TESTING DEV 1 ISOLATION: Simulating network delay instead of actual fetch
        await new Promise((resolve) => setTimeout(resolve, 3000));

        throw new Error("No backend configured yet - returning mock data");

        /* Live Backend Code (Uncomment when Modal is running):
        const response = await fetch('YOUR_MODAL_URL_HERE/analyze', {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'multipart/form-data',
            },
        });

        if (!response.ok) {
            throw new Error('Backend responded with error');
        }

        return await response.json();
        */
    } catch (error) {
        console.log("Returning UI Mock data.");
        // THE DEMO SAVIOR: Gracefully degrade back to Dev 3's mock math coordinates
        return require('../data/mock_response.json');
    }
};
