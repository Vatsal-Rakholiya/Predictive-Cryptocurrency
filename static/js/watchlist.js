document.addEventListener('DOMContentLoaded', function() {
    // Set up the watchlist functionality
    setupWatchlistButtons();
    
    // Set up coin row click
    setupCoinRowClick();
});

// Setup remove from watchlist buttons
function setupWatchlistButtons() {
    const removeButtons = document.querySelectorAll('.remove-from-watchlist');
    
    removeButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation(); // Prevent triggering the coin row click
            
            const coinId = this.getAttribute('data-coin-id');
            
            if (!coinId) {
                return;
            }
            
            // Confirm before removing
            if (confirm('Are you sure you want to remove this coin from your watchlist?')) {
                removeFromWatchlist(coinId, this);
            }
        });
    });
}

// Function to handle removing coins from watchlist
function removeFromWatchlist(coinId, buttonElement) {
    fetch(`/watchlist/remove/${coinId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Find the parent table row and remove it with animation
            const row = buttonElement.closest('tr');
            
            if (row) {
                row.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                row.style.opacity = '0';
                row.style.transform = 'translateX(20px)';
                
                setTimeout(() => {
                    row.remove();
                    
                    // Check if there are any coins left
                    const remainingRows = document.querySelectorAll('.watchlist-table tbody tr');
                    if (remainingRows.length === 0) {
                        // Show empty state
                        const tableContainer = document.querySelector('.table-responsive');
                        if (tableContainer) {
                            tableContainer.innerHTML = `
                                <div class="text-center p-5">
                                    <i class="fas fa-star fa-3x mb-3 text-muted"></i>
                                    <h5>Your watchlist is empty</h5>
                                    <p class="text-muted">Add coins to your watchlist to track them here</p>
                                    <a href="/" class="btn btn-primary mt-2">Browse Market</a>
                                </div>
                            `;
                        }
                    }
                }, 300);
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
}

// Setup coin row click to navigate to coin details
function setupCoinRowClick() {
    const coinRows = document.querySelectorAll('.coin-row');
    
    coinRows.forEach(row => {
        row.addEventListener('click', function() {
            const coinId = this.getAttribute('data-coin-id');
            if (coinId) {
                window.location.href = `/coin/${coinId}`;
            }
        });
    });
}
