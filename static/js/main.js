document.addEventListener('DOMContentLoaded', function() {
    
    // Theme toggling functionality
    const body = document.body;
    const themeToggle = document.getElementById('theme-toggle');
    
    if (themeToggle) {
        themeToggle.addEventListener('click', function() {
            // Toggle dark mode class on body
            body.classList.toggle('dark-mode');
            
            // Update icon
            const icon = themeToggle.querySelector('i');
            if (body.classList.contains('dark-mode')) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            } else {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
            
            // Save preference for logged in users via AJAX
            fetch('/toggle-theme', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            })
            .then(response => response.json())
            .catch(error => console.error('Error:', error));
        });
        
        // Set initial icon based on current mode
        const icon = themeToggle.querySelector('i');
        if (body.classList.contains('dark-mode')) {
            icon.classList.remove('fa-moon');
            icon.classList.add('fa-sun');
        }
    }
    
    // Coin row click handler
    const coinRows = document.querySelectorAll('.coin-row');
    coinRows.forEach(row => {
        row.addEventListener('click', function() {
            const coinId = this.getAttribute('data-coin-id');
            if (coinId) {
                window.location.href = `/coin/${coinId}`;
            }
        });
    });
    
    // Trending coin click handler
    const trendingCoins = document.querySelectorAll('.trending-coin');
    trendingCoins.forEach(coin => {
        coin.addEventListener('click', function() {
            const coinId = this.getAttribute('data-coin-id');
            if (coinId) {
                window.location.href = `/coin/${coinId}`;
            }
        });
    });
    
    // Search functionality
    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');
    const searchResults = document.getElementById('search-results');
    
    let searchTimeout = null;
    
    if (searchInput) {
        // Search input handler
        searchInput.addEventListener('input', function() {
            const query = this.value.trim();
            
            // Clear previous timeout
            if (searchTimeout) {
                clearTimeout(searchTimeout);
            }
            
            // Hide results if query is empty
            if (query.length < 2) {
                searchResults.classList.remove('active');
                searchResults.innerHTML = '';
                return;
            }
            
            // Set timeout to avoid too many requests
            searchTimeout = setTimeout(() => {
                fetchSearchResults(query);
            }, 300);
        });
        
        // Search button handler
        if (searchButton) {
            searchButton.addEventListener('click', function() {
                const query = searchInput.value.trim();
                if (query.length >= 2) {
                    fetchSearchResults(query);
                }
            });
        }
        
        // Close search results when clicking outside
        document.addEventListener('click', function(event) {
            if (!searchInput.contains(event.target) && !searchResults.contains(event.target)) {
                searchResults.classList.remove('active');
            }
        });
    }
    
    // Function to fetch search results
    function fetchSearchResults(query) {
        fetch(`/search?q=${encodeURIComponent(query)}`)
            .then(response => response.json())
            .then(data => {
                renderSearchResults(data);
            })
            .catch(error => {
                console.error('Error searching:', error);
            });
    }
    
    // Function to render search results
    function renderSearchResults(results) {
        searchResults.innerHTML = '';
        
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
            searchResults.classList.add('active');
            return;
        }
        
        results.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.className = 'search-result-item';
            resultItem.innerHTML = `
                <div class="d-flex align-items-center">
                    <img src="${result.thumb}" alt="${result.name}" class="coin-icon me-2">
                    <div>
                        <span class="coin-name">${result.name}</span>
                        <span class="coin-symbol text-muted">${result.symbol}</span>
                    </div>
                    <span class="ms-auto badge bg-secondary">#${result.market_cap_rank || 'N/A'}</span>
                </div>
            `;
            
            resultItem.addEventListener('click', function() {
                window.location.href = `/coin/${result.id}`;
            });
            
            searchResults.appendChild(resultItem);
        });
        
        searchResults.classList.add('active');
    }
    
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});
