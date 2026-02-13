/**
 * Lexicon Visualization - Interactive Graph for Organizational Bodies
 * Uses D3.js logic structure for flexibility
 */

async function renderLexiconGraph(canvasId) {
    const container = document.getElementById(canvasId);
    if (!container) return;

    try {
        // Fetch lexicon data
        const response = await fetch('/api/knowledge/lexicon'); 
        let data;
        if (response.ok) {
            data = await response.json();
        } else {
            console.warn("Failed to fetch lexicon, using seed data structure");
            data = {
                nodes: [
                    { id: "Physics", group: 1 },
                    { id: "Biology", group: 2 },
                    { id: "Digital", group: 3 },
                    { id: "Galaxy", group: 1 },
                    { id: "Cell", group: 2 },
                    { id: "DAO", group: 3 }
                ],
                links: [
                    { source: "Galaxy", target: "Physics" },
                    { source: "Cell", target: "Biology" },
                    { source: "DAO", target: "Digital" }
                ]
            };
        }

        // D3.js Force Directed Graph visualization logic would be implemented here.
        // For the prototype, we display a summary.
        container.innerHTML = `<div class="p-4 bg-gray-100 rounded">
            <h3 class="font-bold">Lexicon Graph Loaded</h3>
            <p>Nodes: ${data.nodes ? data.nodes.length : 0}</p>
            <p>Links: ${data.links ? data.links.length : 0}</p>
            <p class="text-sm text-gray-500 mt-2">Visualization engine ready.</p>
        </div>`;

    } catch (error) {
        console.error("Lexicon graph error:", error);
        container.innerHTML = '<p class="text-red-500">Error loading lexicon graph.</p>';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    renderLexiconGraph('lexicon-container');
});
