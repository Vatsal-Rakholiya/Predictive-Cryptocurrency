document.addEventListener('DOMContentLoaded', function() {
    // Initialize price chart if data and element exist
    initializePriceChart();
    
    // Setup watchlist button functionality
    setupWatchlistButton();
});

// Initialize the main price chart
function initializePriceChart() {
    const chartCanvas = document.getElementById('priceChart');
    
    if (!chartCanvas || !chartData || !chartData.labels || !chartData.prices) {
        console.error('Missing chart data or element');
        return;
    }
    
    // Determine if overall price trend is positive
    const firstPrice = chartData.prices[0];
    const lastPrice = chartData.prices[chartData.prices.length - 1];
    const isPositiveTrend = lastPrice >= firstPrice;
    
    // Set colors based on trend
    const primaryColor = isPositiveTrend ? '40, 167, 69' : '220, 53, 69'; // rgb format
    const borderColor = isPositiveTrend ? 'rgba(40, 167, 69, 1)' : 'rgba(220, 53, 69, 1)';
    
    // Determine if dark mode is active
    const isDarkMode = document.body.classList.contains('dark-mode');
    const gridColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    const textColor = isDarkMode ? '#e0e0e0' : '#666666';
    
    // Format Y-axis values as currency
    const currencyFormatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: selectedCurrency.toUpperCase(),
        minimumFractionDigits: 2,
        maximumFractionDigits: 6
    });
    
    // Create the chart
    const priceChart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: chartData.labels,
            datasets: [{
                label: `Price (${selectedCurrency.toUpperCase()})`,
                data: chartData.prices,
                borderColor: borderColor,
                backgroundColor: (context) => {
                    const ctx = context.chart.ctx;
                    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
                    gradient.addColorStop(0, `rgba(${primaryColor}, 0.2)`);
                    gradient.addColorStop(1, `rgba(${primaryColor}, 0)`);
                    return gradient;
                },
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBorderWidth: 2,
                pointHoverBackgroundColor: borderColor,
                pointHoverBorderColor: isDarkMode ? '#fff' : '#fff',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: isDarkMode ? 'rgba(30, 30, 30, 0.9)' : 'rgba(255, 255, 255, 0.9)',
                    titleColor: isDarkMode ? '#e0e0e0' : '#333',
                    bodyColor: isDarkMode ? '#e0e0e0' : '#333',
                    borderColor: isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 6,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return currencyFormatter.format(context.parsed.y);
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        maxRotation: 0,
                        autoSkipPadding: 20,
                        color: textColor
                    }
                },
                y: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor,
                        callback: function(value) {
                            return currencyFormatter.format(value);
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuad'
            }
        }
    });
    
    // Resize observer to handle container size changes
    new ResizeObserver(entries => {
        priceChart.resize();
    }).observe(chartCanvas.parentElement);
}

// Setup watchlist button functionality
function setupWatchlistButton() {
    const watchlistBtn = document.getElementById('watchlist-btn');
    
    if (!watchlistBtn) {
        return;
    }
    
    watchlistBtn.addEventListener('click', function(e) {
        e.preventDefault();
        
        const coinId = this.getAttribute('data-coin-id');
        const action = this.getAttribute('data-action');
        
        if (!coinId) {
            return;
        }
        
        const url = action === 'add' 
            ? `/watchlist/add/${coinId}` 
            : `/watchlist/remove/${coinId}`;
        
        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Toggle button state
                if (action === 'add') {
                    watchlistBtn.classList.remove('btn-outline-primary');
                    watchlistBtn.classList.add('btn-danger');
                    watchlistBtn.innerHTML = '<i class="fas fa-star me-1"></i> Remove from Watchlist';
                    watchlistBtn.setAttribute('data-action', 'remove');
                } else {
                    watchlistBtn.classList.remove('btn-danger');
                    watchlistBtn.classList.add('btn-outline-primary');
                    watchlistBtn.innerHTML = '<i class="fas fa-star me-1"></i> Add to Watchlist';
                    watchlistBtn.setAttribute('data-action', 'add');
                }
            } else {
                console.error('Error:', data.message);
                
                // Show error alert
                const alertElement = document.createElement('div');
                alertElement.className = 'alert alert-danger alert-dismissible fade show';
                alertElement.innerHTML = `
                    ${data.message}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                `;
                
                document.querySelector('main').prepend(alertElement);
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
}
