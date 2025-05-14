document.addEventListener('DOMContentLoaded', function() {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const tweetsAnalyzed = document.getElementById('tweetsAnalyzed');
    const misinfoDetected = document.getElementById('misinfoDetected');
    const opinionsDetected = document.getElementById('opinionsDetected');
    
    checkApiConnection();
    
    chrome.storage.local.get(['tweetsAnalyzed', 'misinfoDetected', 'opinionsDetected'], function(result) {
        tweetsAnalyzed.textContent = result.tweetsAnalyzed || 0;
        misinfoDetected.textContent = result.misinfoDetected || 0;
        opinionsDetected.textContent = result.opinionsDetected || 0;
    });
    
    async function checkApiConnection() {
        try {
            const response = await fetch('http://localhost:5000/', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            if (response.ok) {
                statusIndicator.classList.add('connected');
                statusIndicator.classList.remove('disconnected');
                statusText.textContent = 'Connected to API';
            } else {
                throw new Error('API returned error');
            }
        } catch (error) {
            statusIndicator.classList.add('disconnected');
            statusIndicator.classList.remove('connected');
            statusText.textContent = 'API not connected';
            
            console.error('API connection error:', error);
        }
    }
});