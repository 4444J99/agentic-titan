/**
 * Inquiry Charts - Visualizations for Epistemic Signatures
 * Requires Chart.js
 */

async function renderEpistemicRadar(canvasId, sessionId) {
    try {
        const response = await fetch(`/api/inquiry/${sessionId}/epistemic_signature`);
        if (!response.ok) {
            console.error('Failed to fetch epistemic signature:', response.statusText);
            return;
        }
        
        const data = await response.json();

        const config = {
            type: 'radar',
            data: data,
            options: {
                elements: {
                    line: {
                        borderWidth: 3
                    }
                },
                scales: {
                    r: {
                        angleLines: {
                            display: false
                        },
                        suggestedMin: 0,
                        suggestedMax: 1
                    }
                }
            },
        };

        new Chart(document.getElementById(canvasId), config);
    } catch (error) {
        console.error('Error rendering chart:', error);
    }
}

// Auto-initialize if canvas exists
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('epistemicRadar');
    if (canvas) {
        const sessionId = canvas.dataset.sessionId || 'current';
        renderEpistemicRadar('epistemicRadar', sessionId);
    }
});
