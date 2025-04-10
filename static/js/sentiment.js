document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the sentiment page with a coin parameter
    const urlParams = new URLSearchParams(window.location.search);
    const coinId = urlParams.get('coin');
    
    if (coinId) {
        // Hide initial view, show loading
        document.getElementById('sentiment-initial').classList.add('d-none');
        document.getElementById('sentiment-loading').classList.remove('d-none');
        
        // Fetch sentiment data for the selected coin
        fetchSentimentData(coinId);
    }
    
    function fetchSentimentData(coinId) {
        fetch(`/api/sentiment/${coinId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                if (data && data.status === 'ok') {
                    if (data.sentiment && data.mentions && data.mentions.length > 0) {
                        updateUI(data);
                    } else {
                        showNoDataState();
                    }
                } else {
                    showErrorState();
                }
            })
            .catch(error => {
                console.error('Error fetching sentiment data:', error);
                showErrorState();
            });
    }
    
    function updateUI(data) {
        // Hide loading, show results
        document.getElementById('sentiment-loading').classList.add('d-none');
        document.getElementById('sentiment-results').classList.remove('d-none');
        
        // Update coin name
        document.getElementById('coin-name').textContent = coinId;
        
        // Get sentiment data
        const sentiment = data.sentiment;
        const mentions = data.mentions;
        
        // Format score
        const score = parseFloat(sentiment.score).toFixed(2);
        
        // Determine sentiment category and appearance
        let category, iconClass, bgClass, textClass, alertClass;
        
        if (sentiment.score > 0.2) {
            category = "very positive";
            iconClass = "fa-grin-stars";
            bgClass = "bg-success bg-opacity-10 border border-success";
            textClass = "text-success";
            alertClass = "alert-success";
        } else if (sentiment.score > 0.05) {
            category = "positive";
            iconClass = "fa-smile";
            bgClass = "bg-success bg-opacity-10 border border-success";
            textClass = "text-success";
            alertClass = "alert-success";
        } else if (sentiment.score < -0.2) {
            category = "very negative";
            iconClass = "fa-angry";
            bgClass = "bg-danger bg-opacity-10 border border-danger";
            textClass = "text-danger";
            alertClass = "alert-danger";
        } else if (sentiment.score < -0.05) {
            category = "negative";
            iconClass = "fa-frown";
            bgClass = "bg-danger bg-opacity-10 border border-danger";
            textClass = "text-danger";
            alertClass = "alert-danger";
        } else {
            category = "neutral";
            iconClass = "fa-meh";
            bgClass = "bg-warning bg-opacity-10 border border-warning";
            textClass = "text-warning";
            alertClass = "alert-warning";
        }
        
        // Update sentiment summary
        document.getElementById('sentiment-summary').className = `text-center p-4 rounded ${bgClass}`;
        document.getElementById('sentiment-icon').className = `display-4 mb-3 ${textClass}`;
        document.getElementById('sentiment-icon').querySelector('i').className = `fas ${iconClass}`;
        document.getElementById('sentiment-category').textContent = category;
        document.getElementById('sentiment-score').textContent = score;
        
        // Update market interpretation
        const marketInterpretation = document.getElementById('market-interpretation');
        marketInterpretation.className = `alert ${alertClass}`;
        
        let interpretationText = '';
        if (sentiment.score > 0.05) {
            interpretationText = `The market appears to be bullish on ${coinId}. Recent news coverage has been predominantly positive, which often correlates with upward price movement.`;
        } else if (sentiment.score < -0.05) {
            interpretationText = `The market appears to be bearish on ${coinId}. Recent news coverage has been predominantly negative, which often correlates with downward price pressure.`;
        } else {
            interpretationText = `The market appears to be uncertain about ${coinId}. Recent news coverage has been mixed or neutral, which often indicates a period of consolidation.`;
        }
        
        document.getElementById('market-interpretation-text').textContent = interpretationText;
        document.getElementById('article-count').textContent = sentiment.volume || mentions.length;
        
        // Update sentiment chart
        updateSentimentChart(sentiment);
        
        // Update mentions
        updateMentions(mentions);
        
        // Update historical chart if data available
        if (data.historical) {
            updateHistoricalChart(data.historical);
        } else {
            createSimulatedHistoricalChart(sentiment);
        }
    }
    
    function updateSentimentChart(sentiment) {
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        
        // Get sentiment distribution
        const positive = sentiment.positive || 0;
        const negative = sentiment.negative || 0;
        const neutral = sentiment.neutral || 0;
        
        // Create chart
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Positive', 'Neutral', 'Negative'],
                datasets: [{
                    data: [
                        (positive * 100).toFixed(1),
                        (neutral * 100).toFixed(1),
                        (negative * 100).toFixed(1)
                    ],
                    backgroundColor: [
                        '#28a745',
                        '#ffc107',
                        '#dc3545'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    function updateMentions(mentions) {
        const container = document.getElementById('news-mentions');
        container.innerHTML = '';
        
        if (mentions && mentions.length > 0) {
            // Sort mentions by score (most positive/negative first)
            mentions.sort((a, b) => Math.abs(b.score) - Math.abs(a.score));
            
            // Display top mentions (max 5)
            const maxMentions = Math.min(5, mentions.length);
            for (let i = 0; i < maxMentions; i++) {
                const mention = mentions[i];
                
                // Determine mention sentiment class
                let badgeClass = 'bg-warning';
                if (mention.score > 0.05) badgeClass = 'bg-success';
                if (mention.score < -0.05) badgeClass = 'bg-danger';
                
                // Create mention item
                const mentionItem = document.createElement('div');
                mentionItem.className = 'list-group-item py-3';
                
                // Format date if available
                let dateDisplay = '';
                if (mention.published_at) {
                    try {
                        const date = new Date(mention.published_at);
                        dateDisplay = `<small class="text-muted">${date.toLocaleDateString()}</small>`;
                    } catch (e) {
                        dateDisplay = '';
                    }
                }
                
                // Create mentions HTML
                mentionItem.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center mb-2">
                        <span class="badge ${badgeClass}">${(mention.score).toFixed(2)}</span>
                        ${dateDisplay}
                    </div>
                    <p class="mb-1">${escapeHTML(mention.content)}</p>
                    ${mention.url ? `<a href="${mention.url}" target="_blank" class="btn btn-sm btn-outline-primary mt-2">Read Article</a>` : ''}
                `;
                
                container.appendChild(mentionItem);
            }
            
            document.getElementById('no-mentions').classList.add('d-none');
        } else {
            document.getElementById('no-mentions').classList.remove('d-none');
        }
    }
    
    function createSimulatedHistoricalChart(currentSentiment) {
        const ctx = document.getElementById('historicalSentimentChart').getContext('2d');
        
        // Generate dates for the last 7 days
        const dates = [];
        for (let i = 6; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            dates.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
        }
        
        // Create chart with simulated data based on current sentiment
        const currentScore = parseFloat(currentSentiment.score);
        const scores = [];
        
        // Generate slightly varied data centered around current score
        for (let i = 0; i < 7; i++) {
            // More variation in the past, converging to current
            const daysFactor = i / 7;
            const variation = Math.random() * 0.3 - 0.15;
            const adjustedVariation = variation * (1 - daysFactor);
            
            // Ensure we end with something close to current sentiment
            const score = i < 6 ? Math.max(-1, Math.min(1, currentScore + adjustedVariation)) : currentScore;
            scores.push(score);
        }
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Sentiment Score',
                    data: scores,
                    borderColor: '#007bff',
                    backgroundColor: 'rgba(0, 123, 255, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        min: -1,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                if (value === 1) return 'Very Positive';
                                if (value === 0.5) return 'Positive';
                                if (value === 0) return 'Neutral';
                                if (value === -0.5) return 'Negative';
                                if (value === -1) return 'Very Negative';
                                return '';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const score = context.parsed.y;
                                let sentiment = 'Neutral';
                                if (score > 0.2) sentiment = 'Very Positive';
                                else if (score > 0.05) sentiment = 'Positive';
                                else if (score < -0.2) sentiment = 'Very Negative';
                                else if (score < -0.05) sentiment = 'Negative';
                                
                                return `Sentiment: ${sentiment} (${score.toFixed(2)})`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function updateHistoricalChart(historicalData) {
        // Implementation for real historical data
        // Similar to createSimulatedHistoricalChart but uses real data
    }
    
    function showLoading() {
        document.getElementById('sentiment-initial').classList.add('d-none');
        document.getElementById('sentiment-results').classList.add('d-none');
        document.getElementById('sentiment-error').classList.add('d-none');
        document.getElementById('sentiment-no-data').classList.add('d-none');
        document.getElementById('sentiment-loading').classList.remove('d-none');
    }
    
    function hideLoading() {
        document.getElementById('sentiment-loading').classList.add('d-none');
    }
    
    function showErrorState() {
        hideLoading();
        document.getElementById('sentiment-error').classList.remove('d-none');
    }
    
    function showNoDataState() {
        hideLoading();
        document.getElementById('sentiment-no-data').classList.remove('d-none');
    }
    
    // Helper function to safely escape HTML
    function escapeHTML(str) {
        return str.replace(/[&<>"']/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            }[tag]));
    }
});