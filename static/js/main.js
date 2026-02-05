// Update statistics and detections periodically
function updateStats() {
    fetch('/api/statistics')
        .then(response => response.json())
        .then(data => {
            document.getElementById('total-detections').textContent = data.total_detections;
            document.getElementById('unique-plates').textContent = data.unique_plates;
        })
        .catch(error => console.error('Error fetching statistics:', error));
}

function updateDetections() {
    fetch('/api/detections')
        .then(response => response.json())
        .then(data => {
            const detectionsList = document.getElementById('detections-list');
            detectionsList.innerHTML = '';

            // Reverse to show most recent first
            const detections = data.detections.reverse();

            detections.forEach(detection => {
                const item = document.createElement('div');
                item.className = 'detection-item';

                const timestamp = new Date(detection.timestamp).toLocaleString();

                item.innerHTML = `
                    <div class="plate-number">${detection.text}</div>
                    <div class="detection-info">
                        <span class="timestamp">${timestamp}</span>
                        <span class="confidence">Confidence: ${(detection.confidence * 100).toFixed(1)}%</span>
                    </div>
                `;

                detectionsList.appendChild(item);
            });
        })
        .catch(error => console.error('Error fetching detections:', error));
}

// Initial load
updateStats();
updateDetections();

// Update every 2 seconds
setInterval(() => {
    updateStats();
    updateDetections();
}, 2000);
