document.addEventListener('DOMContentLoaded', function() {
    // Set Chart.js default styles
    configureChartDefaults();
    
    // Create sparkline charts
    createSparklineCharts();
    
    // Load prediction directions for all coins
    loadPredictionDirections();
});

// Configure Chart.js global defaults
function configureChartDefaults() {
    const isDarkMode = document.body.classList.contains('dark-mode');
    
    Chart.defaults.color = isDarkMode ? '#e0e0e0' : '#666666';
    Chart.defaults.borderColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // Custom linear gradient for line charts
    const createGradient = (ctx, primaryColor, opacity) => {
        const gradient = ctx.createLinearGradient(0, 0, 0, ctx.canvas.height);
        gradient.addColorStop(0, `rgba(${primaryColor}, ${opacity})`);
        gradient.addColorStop(1, `rgba(${primaryColor}, 0)`);
        return gradient;
    };
    
    // Custom plugin for responsive fonts
    const responsiveFontPlugin = {
        id: 'responsiveFont',
        beforeInit: (chart) => {
            const originalFit = chart.legend.fit;
            chart.legend.fit = function fit() {
                originalFit.call(chart.legend);
                this.height += 10;
            };
        }
    };
    
    // Add plugin globally
    Chart.register(responsiveFontPlugin);
}

// Create sparkline charts
function createSparklineCharts() {
    const sparklineElements = document.querySelectorAll('.sparkline-chart');
    
    sparklineElements.forEach(element => {
        const prices = JSON.parse(element.getAttribute('data-prices'));
        
        if (!prices || prices.length === 0) {
            return;
        }
        
        // Determine if price trend is positive or negative
        const startPrice = prices[0];
        const endPrice = prices[prices.length - 1];
        const isPositive = endPrice >= startPrice;
        
        // Set colors based on trend
        const primaryColor = isPositive ? '40, 167, 69' : '220, 53, 69';
        const lineColor = isPositive ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)';
        
        // Create canvas element
        const canvas = document.createElement('canvas');
        element.appendChild(canvas);
        
        // Create chart
        new Chart(canvas, {
            type: 'line',
            data: {
                labels: Array(prices.length).fill(''),
                datasets: [{
                    data: prices,
                    borderColor: lineColor,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.4,
                    fill: true,
                    backgroundColor: (context) => {
                        const ctx = context.chart.ctx;
                        const gradient = ctx.createLinearGradient(0, 0, 0, 40);
                        gradient.addColorStop(0, `rgba(${primaryColor}, 0.2)`);
                        gradient.addColorStop(1, `rgba(${primaryColor}, 0)`);
                        return gradient;
                    }
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false
                    }
                },
                animation: {
                    duration: 1000,
                    easing: 'easeOutQuad'
                }
            }
        });
    });
}

// Load prediction directions for all coins
function loadPredictionDirections() {
    // Get all coin IDs from the table
    const coinRows = document.querySelectorAll('.coin-row');
    const coinIds = Array.from(coinRows).map(row => row.getAttribute('data-coin-id')).filter(id => id);
    
    if (coinIds.length === 0) {
        return; // No coins to predict
    }
    
    // Start with just top 3 coins to avoid rate limiting
    // Place Ethereum first to ensure it's processed (since it uses CSV data)
    let topCoinIds = coinIds.slice(0, 3);
    
    // If Ethereum is in the list but not in the top 3, add it
    if (coinIds.includes('ethereum') && !topCoinIds.includes('ethereum')) {
        // Replace the last coin with Ethereum
        topCoinIds[topCoinIds.length - 1] = 'ethereum';
    }
    
    // Set initial unknown state for all indicators
    coinIds.forEach(coinId => {
        const predictionCell = document.querySelector(`.prediction-indicator[data-coin-id="${coinId}"]`);
        if (predictionCell) {
            predictionCell.innerHTML = `
                <div class="prediction-indicator-unknown" title="Loading prediction...">
                    <i class="fas fa-sync-alt fa-spin"></i>
                </div>
            `;
        }
    });
    
    // Get the selected currency
    const currencyDropdown = document.getElementById('currencyDropdown');
    const currency = currencyDropdown ? currencyDropdown.textContent.trim().toLowerCase() : 'usd';
    
    // Also get a specialized prediction for Ethereum (using CSV data)
    if (coinIds.includes('ethereum')) {
        loadEthereumPrediction('ethereum', currency);
    }
    
    // Make API call to get predictions for top coins
    fetch(`/api/batch-prediction-directions?coin_ids=${topCoinIds.join(',')}&currency=${currency}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to get predictions');
            }
            return response.json();
        })
        .then(data => {
            // Update UI with prediction indicators
            if (data && data.results) {
                Object.keys(data.results).forEach(coinId => {
                    const predictionData = data.results[coinId];
                    const predictionCell = document.querySelector(`.prediction-indicator[data-coin-id="${coinId}"]`);
                    
                    if (predictionCell) {
                        updatePredictionIndicator(predictionCell, predictionData);
                    }
                });
            }
            
            // Set unknown state for all other coins
            coinIds.forEach(coinId => {
                if (!data.results || !data.results[coinId]) {
                    const predictionCell = document.querySelector(`.prediction-indicator[data-coin-id="${coinId}"]`);
                    if (predictionCell && coinId !== 'ethereum') { // Skip Ethereum as it's handled separately
                        updatePredictionIndicator(predictionCell, {direction: 'unknown', reason: 'not_processed'});
                    }
                }
            });
        })
        .catch(error => {
            console.error('Error loading prediction directions:', error);
            
            // Set unknown state for all coins on error
            coinIds.forEach(coinId => {
                const predictionCell = document.querySelector(`.prediction-indicator[data-coin-id="${coinId}"]`);
                if (predictionCell && coinId !== 'ethereum') { // Skip Ethereum as it's handled separately
                    updatePredictionIndicator(predictionCell, {direction: 'unknown', reason: 'error'});
                }
            });
        });
}

// Load Ethereum prediction from CSV-based endpoint
function loadEthereumPrediction(coinId, currency) {
    const predictionCell = document.querySelector(`.prediction-indicator[data-coin-id="${coinId}"]`);
    if (!predictionCell) return;
    
    fetch(`/api/eth-prediction?currency=${currency}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to get Ethereum prediction');
            }
            return response.json();
        })
        .then(data => {
            if (data && data.prediction_24h) {
                const predictionData = {
                    direction: data.prediction_24h.direction,
                    percent_change: data.prediction_24h.percent_change
                };
                updatePredictionIndicator(predictionCell, predictionData);
            } else {
                updatePredictionIndicator(predictionCell, {direction: 'unknown', reason: 'no_prediction_data'});
            }
        })
        .catch(error => {
            console.error('Error loading Ethereum prediction:', error);
            updatePredictionIndicator(predictionCell, {direction: 'unknown', reason: 'error'});
        });
}

// Update prediction indicator element
function updatePredictionIndicator(element, predictionData) {
    if (!element) return;
    
    if (predictionData.direction === 'up') {
        element.innerHTML = `
            <div class="prediction-indicator-up" title="Predicted to rise in the next 24 hours">
                <i class="fas fa-arrow-up"></i>
                <span class="prediction-percent">+${Math.abs(predictionData.percent_change).toFixed(2)}%</span>
            </div>
        `;
        element.classList.add('text-success');
        element.classList.remove('text-danger', 'text-muted');
    } 
    else if (predictionData.direction === 'down') {
        element.innerHTML = `
            <div class="prediction-indicator-down" title="Predicted to fall in the next 24 hours">
                <i class="fas fa-arrow-down"></i>
                <span class="prediction-percent">-${Math.abs(predictionData.percent_change).toFixed(2)}%</span>
            </div>
        `;
        element.classList.add('text-danger');
        element.classList.remove('text-success', 'text-muted');
    }
    else {
        // Get reason message based on the reason code
        let tooltip = "Prediction unavailable";
        let icon = "fa-minus";
        
        if (predictionData.reason === 'rate_limited' || predictionData.reason === 'rate_limiting_protection') {
            tooltip = "API rate limit reached, try again later";
            icon = "fa-exclamation-circle";
        } else if (predictionData.reason === 'no_model') {
            tooltip = "Prediction model not yet trained for this coin";
            icon = "fa-cog";
        } else if (predictionData.reason === 'error') {
            tooltip = "Error generating prediction";
            icon = "fa-exclamation-triangle";
        } else if (predictionData.reason === 'no_prediction_data') {
            tooltip = "Insufficient data for prediction";
            icon = "fa-chart-line";
        }
        
        element.innerHTML = `
            <div class="prediction-indicator-unknown" title="${tooltip}">
                <i class="fas ${icon}"></i>
            </div>
        `;
        element.classList.add('text-muted');
        element.classList.remove('text-success', 'text-danger');
    }
}
