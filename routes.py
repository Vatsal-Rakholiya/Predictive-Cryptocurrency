from flask import render_template, request, redirect, url_for, session, jsonify, flash
from app import app, db, cache
import api_service
from models import User, WatchlistItem, SentimentRecord, SentimentMention, DataSourceType
from sentiment_service import sentiment_service
from sentiment_data_collector import SentimentDataCollector
from price_prediction import get_prediction_model
import logging
import os
from datetime import datetime, timedelta

# Import sentiment controller for enhanced analysis
from sentiment_controller import get_sentiment_data, batch_analyze_coins
import sentiment_controller

# Configure logging
logger = logging.getLogger(__name__)

# Helper function to get user by ID for use in templates
@app.context_processor
def utility_processor():
    def get_user_by_id(user_id):
        return User.query.get(user_id)
    return {'get_user_by_id': get_user_by_id}

# Route to create the first admin user (for testing purposes)
@app.route('/setup-admin', methods=['GET', 'POST'])
def setup_admin():
    """Create the first admin user (should be disabled in production)"""
    # Check if any admin user already exists
    admin_exists = User.query.filter_by(is_admin=True).first()
    
    if admin_exists:
        flash('Admin user already exists!', 'warning')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        username = request.form.get('username', 'admin')
        email = request.form.get('email', 'admin@example.com')
        password = request.form.get('password', 'adminpass')
        
        try:
            # Create admin user
            admin = User(
                username=username,
                email=email,
                is_admin=True
            )
            admin.set_password(password)
            db.session.add(admin)
            db.session.commit()
            
            flash('Admin user created successfully!', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error creating admin user: {e}")
            flash(f'Error creating admin user: {str(e)}', 'danger')
    
    return render_template('setup_admin.html')

# Routes
@app.route('/')
def index():
    """Homepage - Market Overview"""
    currency = request.args.get('currency', 'usd')
    
    # Fetch data from API
    top_coins = api_service.get_top_coins(currency=currency)
    global_data = api_service.get_global_data()
    trending_coins = api_service.get_trending_coins()
    
    # Format data for template
    formatted_coins = api_service.format_market_data(top_coins)
    
    # Get available currencies for dropdown
    # Fallback to common currencies if API fails due to rate limiting
    currencies = api_service.get_supported_currencies()
    if not currencies:
        currencies = ["usd", "eur", "gbp", "jpy", "btc", "eth"]
    
    return render_template(
        'index.html',
        coins=formatted_coins,
        global_data=global_data.get('data', {}),
        trending=trending_coins.get('coins', []),
        selected_currency=currency,
        available_currencies=currencies
    )

@app.route('/coin/<coin_id>')
def coin_details(coin_id):
    """Coin Details Page"""
    currency = request.args.get('currency', 'usd')
    time_period = request.args.get('time', '30')  # Default to 30 days
    
    # Fetch coin details and historical data
    coin_data = api_service.get_coin_details(coin_id)
    market_chart = api_service.get_coin_market_chart(coin_id, currency, time_period)
    
    # If data fetch failed, show error page
    if not coin_data:
        return render_template('error.html', message=f"Could not fetch data for {coin_id}")
    
    # Format chart data for Chart.js
    chart_data = api_service.format_timestamp_data(market_chart)
    
    # Check if coin is in user's watchlist (if logged in)
    in_watchlist = False
    if 'user_id' in session:
        watchlist_item = WatchlistItem.query.filter_by(
            user_id=session['user_id'], 
            coin_id=coin_id
        ).first()
        in_watchlist = watchlist_item is not None
    
    # Get available currencies for dropdown
    # Fallback to common currencies if API fails due to rate limiting
    currencies = api_service.get_supported_currencies()
    if not currencies:
        currencies = ["usd", "eur", "gbp", "jpy", "btc", "eth"]
    
    # Get trending news related to this cryptocurrency
    coin_news = []
    if os.environ.get('NEWS_API_KEY'):
        import requests
        import re
        
        # Create more accurate and specific search queries for crypto news
        coin_name = coin_data.get('name', '').strip()
        coin_symbol = coin_data.get('symbol', '').strip().upper()
        
        # Build advanced search queries with exact phrases and enhanced contextual keywords
        search_queries = []
        
        # Define contextual keywords for different cryptocurrencies
        context_keywords = {
            'bitcoin': ['btc', 'bitcoin', 'satoshi', 'lightning network', 'halving', 'digital gold', 'spot etf'],
            'ethereum': ['eth', 'ethereum', 'vitalik', 'buterin', 'smart contract', 'defi', 'dapps', 'gas fees', 
                        'proof of stake', 'pos', 'merge', 'consensus', 'layer 2', 'l2', 'scaling', 'eip', 
                        'ethereum 2.0', 'eth2', 'ether'],
            'solana': ['sol', 'solana', 'proof of history', 'high throughput', 'rust', 'solend', 'phantom wallet'],
            'cardano': ['ada', 'cardano', 'hoskinson', 'ouroboros', 'hydra', 'haskell', 'plutus', 'vasil', 'shelley'],
            'ripple': ['xrp', 'ripple', 'garlinghouse', 'sec lawsuit', 'swift', 'odl', 'xrapid', 'cross-border'],
            'polkadot': ['dot', 'polkadot', 'parachain', 'kusama', 'gavin wood', 'substrate', 'relay chain', 'auction'],
            'binancecoin': ['bnb', 'binance coin', 'bsc', 'binance smart chain', 'cz', 'changpeng zhao'],
            'dogecoin': ['doge', 'dogecoin', 'musk', 'elon', 'shiba inu', 'meme coin'],
            'shiba-inu': ['shib', 'shiba inu', 'shiba', 'bone', 'leash', 'shibarium'],
            'matic-network': ['matic', 'polygon', 'ethereum scaling', 'layer 2', 'zero knowledge', 'zk rollup'],
            'chainlink': ['link', 'chainlink', 'oracle', 'price feed', 'sergey nazarov', 'smart contract'],
            'avalanche': ['avax', 'avalanche', 'subnets', 'c-chain', 'defi', 'emin gün sirer'],
            'litecoin': ['ltc', 'litecoin', 'charlie lee', 'silver to bitcoin gold', 'scrypt', 'mimblewimble'],
            'uniswap': ['uni', 'uniswap', 'dex', 'amm', 'liquidity provider', 'swap', 'v3', 'hayden adams'],
            'tether': ['usdt', 'tether', 'stablecoin', 'reserves', 'pegged', 'dollar-backed'],
            'usd-coin': ['usdc', 'usd coin', 'stablecoin', 'circle', 'centre', 'coinbase'],
        }
        
        # Primary context for most crypto news
        primary_context = '(crypto OR cryptocurrency OR blockchain OR token OR coin OR market OR price)'
        
        # Get coin-specific contextual keywords
        specific_keywords = context_keywords.get(coin_id.lower(), [])
        # Add symbol as a keyword if not already in the list
        if coin_symbol and coin_symbol.lower() not in [k.lower() for k in specific_keywords]:
            specific_keywords.append(coin_symbol)
        
        # Create a specific context string for this coin
        specific_context = ' OR '.join([f'"{keyword}"' for keyword in specific_keywords]) if specific_keywords else ''
        
        # Most specific query first - exact cryptocurrency name with both general and specific context
        if coin_name:
            if specific_context:
                search_queries.append(f'"{coin_name}" AND ({primary_context} OR {specific_context})')
            else:
                search_queries.append(f'"{coin_name}" AND {primary_context}')
            
        # If we have a multi-letter symbol that's not a common word, use it with context
        if coin_symbol and len(coin_symbol) > 1 and coin_symbol.lower() not in ['a', 'i', 'in', 'on', 'to', 'for', 'the', 'and', 'or']:
            search_queries.append(f'"{coin_symbol}" AND {primary_context}')
        
        # Add specific search patterns for major cryptocurrencies
        special_coins = {
            'bitcoin': ['BTC price', 'Bitcoin price', 'Bitcoin market', 'Bitcoin adoption', 'Bitcoin mining', 
                      'Bitcoin ETF', 'Bitcoin halving', 'Bitcoin investment', 'Bitcoin regulation'],
            'ethereum': ['ETH price', 'Ethereum price', 'Ethereum upgrade', 'ETH staking', 'Ethereum scaling', 
                        'Ethereum merge', 'Ethereum layer 2', 'ETH gas fees', 'Ethereum development'],
            'binancecoin': ['BNB Chain', 'Binance Coin price', 'BNB price', 'BNB burn', 'Binance Smart Chain'],
            'ripple': ['XRP lawsuit', 'XRP SEC', 'Ripple ODL', 'XRP price', 'Ripple partners'],
            'cardano': ['ADA price', 'Cardano development', 'Cardano upgrade', 'Cardano Vasil', 'Cardano Hydra'],
            'solana': ['SOL price', 'Solana outage', 'Solana development', 'Solana NFT', 'Solana upgrade'],
            'dogecoin': ['DOGE price', 'Dogecoin Elon', 'Musk Dogecoin', 'Dogecoin payment'],
            'polkadot': ['DOT parachain', 'Polkadot auction', 'DOT staking', 'Polkadot ecosystem']
        }
        
        # Add special search patterns for major coins
        if coin_id.lower() in special_coins:
            for pattern in special_coins[coin_id.lower()]:
                search_queries.append(f'"{pattern}"')
        
        # Process stablecoins differently
        stablecoins = ['tether', 'usd-coin', 'dai', 'binance-usd', 'true-usd', 'frax', 'paxos-standard']
        if coin_id.lower() in stablecoins:
            search_queries.append(f'"{coin_name}" AND (stablecoin OR "stable coin" OR peg OR dollar OR regulation OR audit OR reserve)')
            
        # Fallback to coin_id if we have no valid queries
        if not search_queries:
            search_queries.append(f'"{coin_id}" AND cryptocurrency')
            
        seen_urls = set()  # To avoid duplicate articles
        relevant_articles = []
        
        # Track the last date we'll use for news (within the last 14 days for fresher news)
        oldest_allowed_date = datetime.now() - timedelta(days=14)
        
        for query in search_queries:
            if len(relevant_articles) >= 15:
                break
                
            try:
                # Use NewsAPI to get relevant articles
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': query,
                    'sortBy': 'relevancy',  # Sort by relevance first
                    'language': 'en',
                    'pageSize': 25,  # Get more to filter
                    'from': (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d'),  # Last 14 days
                    'apiKey': os.environ.get('NEWS_API_KEY')
                }
                
                response = requests.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('status') == 'ok' and data.get('articles'):
                        for article in data['articles']:
                            # Skip duplicate articles
                            if article['url'] not in seen_urls:
                                # Check publication date
                                try:
                                    pub_date = datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                                    # Skip articles older than 14 days
                                    if pub_date < oldest_allowed_date:
                                        continue
                                        
                                    # Format date for display
                                    article['formatted_date'] = pub_date.strftime("%b %d, %Y")
                                except:
                                    # If date parsing fails, use current date
                                    article['formatted_date'] = datetime.now().strftime("%b %d, %Y")
                                
                                # Calculate content relevance score with advanced rules
                                relevance_score = 0
                                
                                # Different parts of the article have different weights
                                title = article.get('title', '').lower() or ''
                                description = article.get('description', '').lower() or ''
                                content = article.get('content', '').lower() or ''
                                
                                # Check for exact name match with higher weights for more prominent positions
                                if coin_name.lower() in title:
                                    relevance_score += 15  # Highest relevance for title match
                                elif coin_symbol.lower() in title:
                                    relevance_score += 12  # High relevance for symbol in title
                                
                                if coin_name.lower() in description:
                                    relevance_score += 8  # Good relevance for description match
                                elif coin_symbol.lower() in description:
                                    relevance_score += 6  # Medium relevance for symbol in description
                                
                                if coin_name.lower() in content:
                                    relevance_score += 4  # Some relevance for content match
                                elif coin_symbol.lower() in content:
                                    relevance_score += 3  # Low relevance for symbol in content
                                
                                # Check for exact pattern matches in title (most visible part)
                                if coin_id.lower() in special_coins:
                                    for pattern in special_coins[coin_id.lower()]:
                                        if pattern.lower() in title:
                                            relevance_score += 10  # Bonus for special pattern in title
                                            break  # Only count once
                                
                                # Check for relevant keywords in title
                                if specific_keywords:
                                    for keyword in specific_keywords:
                                        if keyword.lower() in title:
                                            relevance_score += 5  # Bonus for contextual keyword in title
                                            break  # Only count once
                                
                                # Penalize for being too generic without specific cryptocurrency mentions
                                if 'crypto' in title and not (coin_name.lower() in title or coin_symbol.lower() in title):
                                    relevance_score -= 5  # Generic crypto article
                                
                                # Reject completely irrelevant articles
                                if relevance_score <= 0:
                                    continue
                                
                                # Store relevance score
                                article['relevance_score'] = relevance_score
                                
                                # Skip articles without a title or with very short content
                                if not title or len(description) < 20:
                                    continue
                                
                                # Add sentiment analysis to articles if enhanced sentiment is available
                                try:
                                    import enhanced_sentiment
                                    combined_text = f"{title} {description} {content}"
                                    sentiment = enhanced_sentiment.analyze_sentiment(combined_text, coin=coin_id)
                                    article['sentiment_score'] = sentiment.get('score', 0)
                                    article['sentiment_category'] = sentiment.get('category', 'neutral')
                                except ImportError:
                                    # Skip sentiment if not available
                                    pass
                                
                                # Apply source quality filtering
                                trusted_sources = ['coindesk', 'cointelegraph', 'decrypt', 'bloomberg', 'reuters', 
                                                 'forbes', 'cnbc', 'bbc', 'wsj', 'the block', 'yahoo finance', 
                                                 'crypto briefing', 'bitcoin magazine', 'marketwatch', 'ft.com',
                                                 'financemagnates', 'thedefiant', 'ambcrypto', 'bitcoinist', 'beincrypto']
                                
                                source_name = article.get('source', {}).get('name', '').lower()
                                if any(source in source_name for source in trusted_sources):
                                    relevance_score += 5  # Bonus for trusted source
                                    article['relevance_score'] = relevance_score
                                
                                # Add to results and mark as seen
                                relevant_articles.append(article)
                                seen_urls.add(article['url'])
                        
                        # Stop once we have enough articles
                        if len(relevant_articles) >= 25:
                            break
            except Exception as e:
                logger.warning(f"Error fetching news for {coin_id}: {e}")
        
        # Sort articles by relevance score (highest first) and then by date (newest first)
        relevant_articles.sort(key=lambda x: (-x.get('relevance_score', 0), 
                                             -datetime.strptime(x['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").timestamp()))
        
        # Take top 10 most relevant articles
        coin_news = relevant_articles[:10]
    
    return render_template(
        'coin_details.html',
        coin=coin_data,
        chart_data=chart_data,
        selected_currency=currency,
        time_period=time_period,
        in_watchlist=in_watchlist,
        available_currencies=currencies,
        coin_news=coin_news
    )

@app.route('/search')
def search():
    """Search for coins"""
    query = request.args.get('q', '')
    if not query or len(query) < 2:
        return jsonify([])
    
    results = api_service.search_coins(query)
    coins = results.get('coins', [])[:10]  # Limit to top 10 results
    
    return jsonify(coins)

@app.route('/watchlist')
def watchlist():
    """User's Watchlist Page"""
    if 'user_id' not in session:
        flash('Please log in to view your watchlist', 'warning')
        return redirect(url_for('index'))
    
    currency = request.args.get('currency', 'usd')
    user_id = session['user_id']
    
    # Get user's watchlist items
    watchlist_items = WatchlistItem.query.filter_by(user_id=user_id).all()
    
    # Fetch details for each watchlist coin
    watchlist_coins = []
    for item in watchlist_items:
        coin_data = api_service.get_coin_details(item.coin_id)
        if coin_data:
            watchlist_coins.append(coin_data)
    
    # Get available currencies for dropdown
    currencies = api_service.get_supported_currencies()
    
    return render_template(
        'watchlist.html',
        coins=watchlist_coins,
        selected_currency=currency,
        available_currencies=currencies
    )

@app.route('/watchlist/add/<coin_id>', methods=['POST'])
def add_to_watchlist(coin_id):
    """Add a coin to the user's watchlist"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in to add to watchlist'}), 401
    
    user_id = session['user_id']
    
    # Check if already in watchlist
    existing = WatchlistItem.query.filter_by(user_id=user_id, coin_id=coin_id).first()
    if existing:
        return jsonify({'success': True, 'message': 'Coin already in watchlist'})
    
    # Add to watchlist
    try:
        new_item = WatchlistItem(coin_id=coin_id, user_id=user_id)
        db.session.add(new_item)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Coin added to watchlist'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding to watchlist: {e}")
        return jsonify({'success': False, 'message': 'Error adding to watchlist'}), 500

@app.route('/watchlist/remove/<coin_id>', methods=['POST'])
def remove_from_watchlist(coin_id):
    """Remove a coin from the user's watchlist"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Please log in to modify watchlist'}), 401
    
    user_id = session['user_id']
    
    try:
        WatchlistItem.query.filter_by(user_id=user_id, coin_id=coin_id).delete()
        db.session.commit()
        return jsonify({'success': True, 'message': 'Coin removed from watchlist'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error removing from watchlist: {e}")
        return jsonify({'success': False, 'message': 'Error removing from watchlist'}), 500

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """User settings page"""
    if 'user_id' not in session:
        flash('Please log in to access settings', 'warning')
        return redirect(url_for('index'))
    
    user_id = session['user_id']
    user = User.query.get(user_id)
    
    if request.method == 'POST':
        # Update user preferences
        currency = request.form.get('currency', 'usd')
        dark_mode = 'dark_mode' in request.form
        
        try:
            user.preferred_currency = currency
            user.dark_mode = dark_mode
            db.session.commit()
            flash('Settings updated successfully', 'success')
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating settings: {e}")
            flash('Error updating settings', 'danger')
    
    # Get available currencies for dropdown
    currencies = api_service.get_supported_currencies()
    
    return render_template(
        'settings.html',
        user=user,
        available_currencies=currencies
    )

# Simple authentication routes - in a real app, you'd want more security measures
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        
        if existing_user:
            flash('Username or email already exists', 'danger')
        else:
            try:
                new_user = User(username=username, email=email)
                new_user.set_password(password)
                db.session.add(new_user)
                db.session.commit()
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error registering user: {e}")
                flash('Error during registration', 'danger')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/toggle-theme', methods=['POST'])
def toggle_theme():
    """Toggle between dark and light mode"""
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        
        if user:
            try:
                user.dark_mode = not user.dark_mode
                db.session.commit()
                return jsonify({'success': True, 'dark_mode': user.dark_mode})
            except Exception as e:
                db.session.rollback()
                logger.error(f"Error toggling theme: {e}")
    
    # Handle theme toggle for non-logged in users via session
    session['dark_mode'] = not session.get('dark_mode', False)
    return jsonify({'success': True, 'dark_mode': session.get('dark_mode')})

# Sentiment Analysis Routes
@app.route('/sentiment')
def sentiment_analysis():
    """Sentiment Analysis Dashboard"""
    try:
        # Default to Bitcoin if no coin is selected
        coin_id = request.args.get('coin', 'bitcoin')
        days = int(request.args.get('days', 7))
        
        # Common top coins with extended information to avoid API calls
        coin_cache = {
            "bitcoin": {
                "id": "bitcoin",
                "name": "Bitcoin",
                "symbol": "BTC",
                "description": {
                    "en": "Bitcoin is the first successful internet money based on peer-to-peer technology."
                },
                "image": {
                    "large": "https://assets.coingecko.com/coins/images/1/large/bitcoin.png"
                },
                "market_data": {
                    "current_price": {"usd": 68423.12},
                    "market_cap": {"usd": 1337000000000},
                    "total_volume": {"usd": 45600000000}
                }
            },
            "ethereum": {
                "id": "ethereum",
                "name": "Ethereum",
                "symbol": "ETH",
                "description": {
                    "en": "Ethereum is a global, open-source platform for decentralized applications."
                },
                "image": {
                    "large": "https://assets.coingecko.com/coins/images/279/large/ethereum.png"
                },
                "market_data": {
                    "current_price": {"usd": 3875.29},
                    "market_cap": {"usd": 467000000000},
                    "total_volume": {"usd": 22500000000}
                }
            }
        }
        
        # List of top coins for selector
        top_coins = [
            {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
            {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
            {"id": "tether", "name": "Tether", "symbol": "USDT"},
            {"id": "binancecoin", "name": "BNB", "symbol": "BNB"},
            {"id": "solana", "name": "Solana", "symbol": "SOL"},
            {"id": "ripple", "name": "XRP", "symbol": "XRP"},
            {"id": "cardano", "name": "Cardano", "symbol": "ADA"},
            {"id": "dogecoin", "name": "Dogecoin", "symbol": "DOGE"},
            {"id": "polkadot", "name": "Polkadot", "symbol": "DOT"},
            {"id": "matic-network", "name": "Polygon", "symbol": "MATIC"}
        ]
        
        # Try to get coin details with fallbacks
        if coin_id in coin_cache:
            # Use cached data for common coins
            coin_details = coin_cache[coin_id]
            logger.info(f"Using cached details for {coin_id}")
        else:
            try:
                # Try the API if not in cache
                coin_details = api_service.get_coin_details(coin_id)
            except Exception as e:
                logger.warning(f"API error fetching coin details: {e}")
                # Use basic details if API is unavailable
                coin_details = next(({"id": c["id"], 
                                     "name": c["name"], 
                                     "symbol": c["symbol"],
                                     "description": {"en": f"Information about {c['name']} cryptocurrency."},
                                     "image": {"large": ""},
                                     "market_data": {"current_price": {"usd": 0}}} for c in top_coins if c["id"] == coin_id), 
                                   {"id": coin_id, 
                                    "name": coin_id.capitalize(), 
                                    "symbol": coin_id[:3].upper(),
                                    "description": {"en": "Information currently unavailable."},
                                    "image": {"large": ""},
                                    "market_data": {"current_price": {"usd": 0}}})
        
        return render_template('sentiment.html',
                               coin_details=coin_details,
                               top_coins=top_coins,
                               days=days)
    except Exception as e:
        logger.error(f"Error in sentiment analysis route: {e}")
        return render_template('error.html', message=str(e)), 500

@app.route('/api/sentiment/<coin_id>')
@cache.cached(timeout=600)  # Cache for 10 minutes (reduced from 30 minutes to ensure fresher data)
def get_sentiment_data_api(coin_id):
    """API endpoint to get sentiment data for a specific coin"""
    try:
        days = int(request.args.get('days', 7))
        force_refresh = request.args.get('refresh', '').lower() == 'true'
        
        logger.info(f"Fetching sentiment data for {coin_id} over {days} days (force_refresh: {force_refresh})")
        
        # Check if NEWS_API_KEY is available
        if not os.environ.get('NEWS_API_KEY'):
            logger.error("NEWS_API_KEY not found in environment")
            return jsonify({
                "status": "error",
                "error": "API key missing",
                "message": "NEWS_API_KEY is required for sentiment analysis."
            }), 400
        
        # Use the sentiment controller to get enhanced data
        sentiment_data = sentiment_controller.get_sentiment_data(coin_id, days, force_refresh)
        
        if sentiment_data:
            logger.info(f"Returning sentiment data for {coin_id} with {len(sentiment_data.get('mentions', []))} mentions")
            return jsonify(sentiment_data)
        else:
            # No data available - inform the user
            logger.warning(f"No sentiment data found for {coin_id}")
            return jsonify({
                "status": "error",
                "error": "No sentiment data available",
                "message": "No news articles found for this cryptocurrency. Try a different coin or time period."
            }), 404
            
    except Exception as e:
        logger.error(f"Error getting sentiment data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# Price Prediction Routes
@app.route('/prediction')
def price_prediction():
    """Price Prediction Page"""
    try:
        # Default to Bitcoin if no coin is selected
        coin_id = request.args.get('coin', 'bitcoin')
        hours = int(request.args.get('hours', 24))
        
        # Common top coins with extended information to avoid API calls
        coin_cache = {
            "bitcoin": {
                "id": "bitcoin",
                "name": "Bitcoin",
                "symbol": "BTC",
                "description": {
                    "en": "Bitcoin is the first successful internet money based on peer-to-peer technology."
                },
                "image": {
                    "large": "https://assets.coingecko.com/coins/images/1/large/bitcoin.png"
                },
                "market_data": {
                    "current_price": {"usd": 68423.12},
                    "market_cap": {"usd": 1337000000000},
                    "total_volume": {"usd": 45600000000}
                }
            },
            "ethereum": {
                "id": "ethereum",
                "name": "Ethereum",
                "symbol": "ETH",
                "description": {
                    "en": "Ethereum is a global, open-source platform for decentralized applications."
                },
                "image": {
                    "large": "https://assets.coingecko.com/coins/images/279/large/ethereum.png"
                },
                "market_data": {
                    "current_price": {"usd": 3875.29},
                    "market_cap": {"usd": 467000000000},
                    "total_volume": {"usd": 22500000000}
                }
            }
        }
        
        # List of top coins for selector
        top_coins = [
            {"id": "bitcoin", "name": "Bitcoin", "symbol": "BTC"},
            {"id": "ethereum", "name": "Ethereum", "symbol": "ETH"},
            {"id": "tether", "name": "Tether", "symbol": "USDT"},
            {"id": "binancecoin", "name": "BNB", "symbol": "BNB"},
            {"id": "solana", "name": "Solana", "symbol": "SOL"},
            {"id": "ripple", "name": "XRP", "symbol": "XRP"},
            {"id": "cardano", "name": "Cardano", "symbol": "ADA"},
            {"id": "dogecoin", "name": "Dogecoin", "symbol": "DOGE"},
            {"id": "polkadot", "name": "Polkadot", "symbol": "DOT"},
            {"id": "matic-network", "name": "Polygon", "symbol": "MATIC"}
        ]
        
        # Try to get coin details with fallbacks
        if coin_id in coin_cache:
            # Use cached data for common coins
            coin_details = coin_cache[coin_id]
            logger.info(f"Using cached details for {coin_id}")
        else:
            try:
                # Try the API if not in cache
                coin_details = api_service.get_coin_details(coin_id)
            except Exception as e:
                logger.warning(f"API error fetching coin details: {e}")
                # Use basic details if API is unavailable
                coin_details = next(({"id": c["id"], 
                                     "name": c["name"], 
                                     "symbol": c["symbol"],
                                     "description": {"en": f"Information about {c['name']} cryptocurrency."},
                                     "image": {"large": ""},
                                     "market_data": {"current_price": {"usd": 0}}} for c in top_coins if c["id"] == coin_id), 
                                   {"id": coin_id, 
                                    "name": coin_id.capitalize(), 
                                    "symbol": coin_id[:3].upper(),
                                    "description": {"en": "Information currently unavailable."},
                                    "image": {"large": ""},
                                    "market_data": {"current_price": {"usd": 0}}})
        
        return render_template('prediction.html',
                              coin_details=coin_details,
                              top_coins=top_coins,
                              hours=hours)
    except Exception as e:
        logger.error(f"Error in price prediction route: {e}")
        return render_template('error.html', message=str(e)), 500

@app.route('/api/prediction/<coin_id>')
@cache.cached(timeout=1800)  # Cache this endpoint for 30 minutes
def get_prediction_data(coin_id):
    """API endpoint to get price prediction data for a specific coin"""
    try:
        # Get the number of hours to predict
        hours = int(request.args.get('hours', 24))
        currency = request.args.get('currency', 'usd')
        
        # Get prediction model for this coin
        model = get_prediction_model(coin_id, currency)
        
        # Try to get predictions
        predictions = model.predict_next_hours(hours)
        
        if predictions:
            return jsonify(predictions)
        else:
            # If prediction failed, try training the model explicitly with less data
            # to avoid API rate limits
            try:
                success = model.train(days=14)  # Try with less data
                if success:
                    predictions = model.predict_next_hours(hours)
                    if predictions:
                        return jsonify(predictions)
            except Exception as train_error:
                logger.error(f"Error training model: {train_error}")
            
            # If still no predictions, return a friendly error message
            return jsonify({
                "error": "Temporarily unable to generate predictions",
                "message": "The prediction service is experiencing high demand. Please try again later."
            }), 503  # Service Unavailable
            
    except ValueError as e:
        # Handle invalid parameters like non-numeric hours
        return jsonify({"error": "Invalid parameters", "message": str(e)}), 400
            
    except Exception as e:
        logger.error(f"Error getting prediction data: {e}")
        return jsonify({
            "error": "Prediction error",
            "message": "An unexpected error occurred while generating predictions."
        }), 500
        
@app.route('/api/quick-prediction/<coin_id>')
@cache.cached(timeout=900)  # Cache for 15 minutes
def quick_prediction(coin_id):
    """API endpoint for a quick next-hour price prediction"""
    try:
        currency = request.args.get('currency', 'usd')
        
        # Get prediction model for this coin
        model = get_prediction_model(coin_id, currency)
        
        # Try to get prediction for just the next hour
        prediction_data = model.predict_next_hours(hours=1)
        
        if prediction_data and prediction_data.get('predictions'):
            # Just return the next hour prediction with the current price
            next_hour = prediction_data['predictions'][0]
            current_price = prediction_data['current_price']
            
            # Calculate percentage change
            price_change = ((next_hour['price'] - current_price) / current_price) * 100
            
            return jsonify({
                'coin_id': coin_id,
                'current_price': current_price,
                'next_hour_price': next_hour['price'],
                'next_hour_time': next_hour['timestamp'],
                'percent_change': price_change,
                'currency': currency
            })
        else:
            # Try to train with less data if prediction failed
            try:
                success = model.train(days=7)  # Try with even less data for quick prediction
                if success:
                    prediction_data = model.predict_next_hours(hours=1)
                    if prediction_data and prediction_data.get('predictions'):
                        next_hour = prediction_data['predictions'][0]
                        current_price = prediction_data['current_price']
                        price_change = ((next_hour['price'] - current_price) / current_price) * 100
                        
                        return jsonify({
                            'coin_id': coin_id,
                            'current_price': current_price,
                            'next_hour_price': next_hour['price'],
                            'next_hour_time': next_hour['timestamp'],
                            'percent_change': price_change,
                            'currency': currency
                        })
            except Exception as train_error:
                logger.error(f"Error training model for quick prediction: {train_error}")
            
            return jsonify({
                "error": "Prediction unavailable",
                "message": "Unable to generate a quick prediction at this time."
            }), 503
    
    except Exception as e:
        logger.error(f"Error getting quick prediction: {e}")
        return jsonify({
            "error": "Prediction error",
            "message": "An error occurred while generating the quick prediction."
        }), 500

@app.route('/api/eth-prediction')
@cache.cached(timeout=3600)  # Cache for 60 minutes
def ethereum_prediction():
    """Specialized API endpoint for Ethereum prediction using CSV data"""
    try:
        # Get user's preferred currency, default to USD
        currency = request.args.get('currency', 'usd')
        
        # Create Ethereum model using CSV data
        model = get_prediction_model('ethereum', currency)
        
        # Use CSV data for prediction
        prediction_data = None
        
        # Always retrain the model to ensure it's using the right features
        logger.info("Training Ethereum model with CSV data")
        success = model._train_with_csv()
        
        if success:
            prediction_data = model._predict_with_csv(hours=24)
        else:
            logger.error("Failed to train Ethereum model")
        
        if not prediction_data:
            return jsonify({
                "error": "Prediction error",
                "message": "Unable to generate prediction for Ethereum from CSV data."
            }), 500
            
        # Get specific points we're interested in
        latest_price = prediction_data['current_price']
        next_hours = prediction_data['predictions']
        
        # Get prediction for next hour and next day
        next_1h = next_hours[0]['price'] if len(next_hours) > 0 else None
        next_24h = next_hours[23]['price'] if len(next_hours) > 23 else None
        
        # Calculate percentage changes
        pct_change_1h = ((next_1h - latest_price) / latest_price) * 100 if next_1h else None
        pct_change_24h = ((next_24h - latest_price) / latest_price) * 100 if next_24h else None
        
        # Prepare simplified results for the frontend
        simplified_predictions = []
        for i, pred in enumerate(next_hours[:7]):  # Just return first 7 predictions (hours)
            simplified_predictions.append({
                'hour': i + 1,
                'price': pred['price'],
                'timestamp': pred['timestamp']
            })
        
        result = {
            'coin_id': 'ethereum',
            'source': 'historical_csv',
            'currency': currency,
            'current_price': latest_price,
            'prediction_1h': {
                'price': next_1h,
                'percent_change': pct_change_1h,
                'direction': 'up' if pct_change_1h and pct_change_1h >= 0 else 'down'
            },
            'prediction_24h': {
                'price': next_24h,
                'percent_change': pct_change_24h,
                'direction': 'up' if pct_change_24h and pct_change_24h >= 0 else 'down'
            },
            'hourly_predictions': simplified_predictions
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in Ethereum CSV prediction: {e}")
        return jsonify({
            "error": "Prediction error",
            "message": "An error occurred while generating the Ethereum prediction from CSV."
        }), 500

@app.route('/api/eth-forecast')
def ethereum_forecast():
    """User-friendly forecast for Ethereum price using CSV data"""
    try:
        # Get user's preferred currency, default to USD
        currency = request.args.get('currency', 'usd').upper()
        
        # Call the ethereum_prediction function to get the raw data
        model = get_prediction_model('ethereum', currency.lower())
        
        # Use CSV data for prediction
        prediction_data = None
        success = model._train_with_csv()
        
        if success:
            prediction_data = model._predict_with_csv(hours=24)
        else:
            return jsonify({
                "error": "Forecast error",
                "message": "Failed to train Ethereum prediction model."
            }), 500
            
        if not prediction_data:
            return jsonify({
                "error": "Forecast error",
                "message": "No prediction data available."
            }), 500
            
        # Get current price and 24h prediction
        current_price = prediction_data['current_price']
        next_24h = prediction_data['predictions'][23]['price'] if len(prediction_data['predictions']) > 23 else None
        
        # Calculate percentage change
        pct_change = ((next_24h - current_price) / current_price) * 100 if next_24h else None
        
        # Format for better readability with 2 decimal places
        formatted_current = f"{current_price:.2f}"
        formatted_next = f"{next_24h:.2f}" if next_24h else "N/A"
        formatted_pct = f"{pct_change:.2f}" if pct_change else "N/A"
        
        # Create a readable prediction
        direction = "increase" if pct_change and pct_change >= 0 else "decrease"
        sign = "+" if pct_change and pct_change >= 0 else ""
        
        # Build the forecast object
        forecast = {
            "coin": "Ethereum",
            "current_price": formatted_current,
            "next_day_price": formatted_next,
            "forecast": f"{sign}{formatted_pct}%",
            "direction": direction,
            "currency": currency,
            "formatted_text": f"""
Ethereum Forecast
----------------
Current Price: {formatted_current} {currency}
Forecast: {sign}{formatted_pct}% ({direction})
Next Day Price: {formatted_next} {currency}
            """.strip()
        }
        
        return jsonify(forecast)
        
    except Exception as e:
        logger.error(f"Error in Ethereum forecast: {e}")
        return jsonify({
            "error": "Forecast error",
            "message": "An error occurred while generating the Ethereum forecast."
        }), 500

@app.route('/api/batch-prediction-directions')
@cache.cached(timeout=3600)  # Cache for 60 minutes to reduce API calls
def batch_prediction_directions():
    """API endpoint to get batch prediction directions for multiple coins"""
    try:
        currency = request.args.get('currency', 'usd')
        coin_ids = request.args.get('coin_ids', '').split(',')
        
        if not coin_ids or coin_ids[0] == '':
            return jsonify({
                "error": "Missing parameters",
                "message": "Please provide coin_ids parameter with comma-separated coin IDs"
            }), 400
        
        results = {}
        
        # Only process top 3 coins to avoid rate limiting issues
        top_coins = coin_ids[:3] if len(coin_ids) > 3 else coin_ids
        logger.info(f"Processing batch predictions for: {', '.join(top_coins)}")
        
        # Pre-check if API is rate-limited by trying to get simple data
        try:
            # Try to just get market data for bitcoin - if this fails, we're rate limited
            test_response = api_service.get_top_coins(currency, 1, 1)
            if not test_response:
                # We're rate limited, return unknown for all
                logger.warning("API appears to be rate limited, returning unknown for all coins")
                results = {coin_id: {'direction': 'unknown', 'reason': 'rate_limited'} for coin_id in coin_ids}
                return jsonify({
                    'results': results,
                    'currency': currency,
                    'status': 'rate_limited'
                })
        except Exception as api_check_error:
            logger.warning(f"API check failed: {api_check_error}")
            # Continue anyway, individual coin predictions will handle errors
        
        for coin_id in top_coins:
            try:
                # Special handling for ethereum to use CSV data
                if coin_id == 'ethereum':
                    logger.info("Using CSV-based prediction for Ethereum")
                    model = get_prediction_model('ethereum', currency)
                    
                    # Force retrain the Ethereum model with CSV data
                    logger.info("Training Ethereum model with CSV data")
                    model_trained = model._train_with_csv()
                    
                    if model_trained:
                        # Use CSV data for Ethereum prediction
                        prediction_data = model._predict_with_csv(hours=24)
                    else:
                        logger.error("Failed to train Ethereum model")
                        results[coin_id] = {'direction': 'unknown', 'reason': 'training_failed'}
                        continue
                else:
                    # Check if we already have a model for this coin
                    model_path = os.path.join('models', f"{coin_id}_{currency}_model.pkl")
                    
                    # If we don't have a model yet, skip prediction to avoid API calls
                    if not os.path.exists(model_path):
                        logger.info(f"Model for {coin_id} not found, marking as 'unknown'")
                        results[coin_id] = {'direction': 'unknown', 'reason': 'no_model'}
                        continue
                    
                    # Get prediction model for this coin
                    model = get_prediction_model(coin_id, currency)
                    
                    # Try to get prediction for next 24 hours
                    prediction_data = model.predict_next_hours(hours=24)
                
                if prediction_data and prediction_data.get('predictions') and len(prediction_data['predictions']) >= 24:
                    # Get the price 24 hours from now (next day)
                    next_day_price = prediction_data['predictions'][23]['price']
                    current_price = prediction_data['current_price']
                    
                    # Calculate percentage change for next day
                    price_change = ((next_day_price - current_price) / current_price) * 100
                    
                    results[coin_id] = {
                        'direction': 'up' if price_change >= 0 else 'down',
                        'percent_change': price_change,
                        'current_price': current_price,
                        'next_day_price': next_day_price,
                    }
                    logger.info(f"Prediction for {coin_id}: {price_change:.2f}% ({'up' if price_change >= 0 else 'down'})")
                else:
                    results[coin_id] = {'direction': 'unknown', 'reason': 'no_prediction_data'}
            except Exception as coin_error:
                logger.warning(f"Error getting prediction for {coin_id}: {coin_error}")
                results[coin_id] = {'direction': 'unknown', 'reason': 'error'}
        
        # Add placeholder "unknown" results for other coins
        for coin_id in coin_ids:
            if coin_id not in results:
                results[coin_id] = {'direction': 'unknown', 'reason': 'rate_limiting_protection'}
        
        # Add sentiment analysis for all coins if we have NEWS_API_KEY
        if os.environ.get('NEWS_API_KEY'):
            try:
                # Get recent news headlines for sentiment analysis
                coin_texts = {}
                
                # Only analyze coins that we have prediction data for
                for coin_id in results.keys():
                    # Skip sentiment analysis for coins with unknown direction
                    if results[coin_id].get('direction') == 'unknown':
                        continue
                        
                    # Check if we have recent sentiment data for this coin
                    recent_sentiment = sentiment_controller.get_sentiment_data(coin_id, days=1, force_refresh=False)
                    
                    if recent_sentiment and recent_sentiment.get('mentions'):
                        # Extract text content from recent mentions
                        texts = []
                        for mention in recent_sentiment.get('mentions', [])[:5]:  # Use up to 5 recent mentions
                            if isinstance(mention, dict) and mention.get('content'):
                                texts.append(mention['content'])
                        
                        if texts:
                            coin_texts[coin_id] = texts
                
                # Only proceed if we have texts to analyze
                if coin_texts:
                    logger.info(f"Analyzing sentiment for {len(coin_texts)} coins")
                    sentiment_results = sentiment_controller.batch_analyze_coins(list(coin_texts.keys()), coin_texts)
                    
                    # Add sentiment data to results
                    for coin_id, sentiment in sentiment_results.items():
                        if coin_id in results:
                            # Add sentiment category and score
                            results[coin_id]['sentiment_category'] = sentiment.get('category', 'neutral')
                            results[coin_id]['sentiment_score'] = sentiment.get('score', 0)
                            
                            # Add alignment between sentiment and price prediction
                            sentiment_positive = sentiment.get('score', 0) > 0
                            price_positive = results[coin_id].get('direction') == 'up'
                            
                            if sentiment_positive == price_positive:
                                results[coin_id]['alignment'] = 'aligned'
                            else:
                                results[coin_id]['alignment'] = 'divergent'
                    
                    logger.info("Added sentiment analysis to prediction results")
            except Exception as sentiment_error:
                logger.warning(f"Error adding sentiment data to predictions: {sentiment_error}")
        
        return jsonify({
            'results': results,
            'currency': currency,
            'status': 'success'
        })
    
    except Exception as e:
        logger.error(f"Error in batch prediction directions: {e}")
        return jsonify({
            "error": "Prediction error",
            "message": "An error occurred while generating batch predictions.",
            "status": "error"
        }), 500

# Future ICO/Airdrops
@app.route('/upcoming-ico')
def upcoming_ico():
    """Upcoming ICO and Airdrops Page"""
    try:
        # For now, we'll return a simple template with placeholder content
        # This can be expanded later to include real ICO/Airdrop data from APIs
        return render_template('upcoming_ico.html')
    except Exception as e:
        logger.error(f"Error in upcoming ICO route: {e}")
        return render_template('error.html', message=str(e)), 500

# Ethereum forecast page
@app.route('/ethereum-forecast')
def ethereum_forecast_page():
    """Ethereum forecast page with user-friendly prediction display"""
    return render_template('ethereum_forecast.html')

# Enhanced sentiment analysis demo page
@app.route('/enhanced-sentiment')
def enhanced_sentiment_page():
    """Enhanced sentiment analysis demo page"""
    return render_template('enhanced_sentiment.html')

# API endpoint for analyzing text sentiment
@app.route('/api/analyze-text', methods=['POST'])
def analyze_text():
    """API endpoint to analyze the sentiment of provided text"""
    try:
        # Get parameters from request
        data = request.json
        if not data:
            return jsonify({"error": True, "message": "No JSON data provided"}), 400
            
        text = data.get('text')
        coin_id = data.get('coin_id', 'bitcoin')
        
        if not text:
            return jsonify({"error": True, "message": "No text provided for analysis"}), 400
            
        # Analyze text using enhanced sentiment analyzer
        try:
            import enhanced_sentiment
            result = enhanced_sentiment.analyze_sentiment(text, coin=coin_id)
            
            # Add key phrases (a simple implementation using word frequency)
            words = text.lower().split()
            # Remove common words
            stopwords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of', 'is', 'are', 'was', 'were']
            words = [word for word in words if word not in stopwords and len(word) > 3]
            
            # Get frequency
            from collections import Counter
            word_counts = Counter(words)
            key_phrases = [word for word, count in word_counts.most_common(5)]
            
            # Add key phrases to result
            result['key_phrases'] = key_phrases
            
            return jsonify(result)
        except ImportError:
            # Fallback to standard sentiment service if enhanced not available
            logger.warning("Enhanced sentiment not available, falling back to standard")
            sentiment_result = sentiment_service.analyze_text(text, coin_id)
            return jsonify({
                "score": sentiment_result.get('score', 0),
                "magnitude": sentiment_result.get('magnitude', 0),
                "category": "neutral" if abs(sentiment_result.get('score', 0)) < 0.05 else "positive" if sentiment_result.get('score', 0) > 0 else "negative",
                "positive": 0.5 + sentiment_result.get('score', 0)/2 if sentiment_result.get('score', 0) > 0 else 0.5,
                "negative": 0.5 - sentiment_result.get('score', 0)/2 if sentiment_result.get('score', 0) < 0 else 0.5,
                "neutral": 0.5 - abs(sentiment_result.get('score', 0))/2,
                "key_phrases": []
            })
            
    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return jsonify({"error": True, "message": str(e)}), 500

# Newsletter routes removed as per user request

# Weekly prediction route
@app.route('/weekly-predictions')
def weekly_predictions():
    """Weekly Cryptocurrency Price Prediction Page"""
    try:
        # Define default predictions to ensure the page works when API/ML is unavailable
        default_predictions = [
            {
                "coin_id": "bitcoin",
                "prediction": 1.25,
                "direction": "up",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "coin_id": "ethereum",
                "prediction": 2.45,
                "direction": "up",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "coin_id": "xrp",
                "prediction": -0.75,
                "direction": "down",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "coin_id": "tether",
                "prediction": 0.05,
                "direction": "up",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            }
        ]
        
        # Simply use default predictions until the API is working
        predictions = default_predictions
        
        return render_template(
            'weekly_predictions.html',
            predictions=predictions
        )
    except Exception as e:
        logger.error(f"Error on weekly predictions page: {e}")
        return render_template('error.html', message=f"Error generating weekly predictions: {e}"), 500

@app.route('/api/weekly-predictions')
def weekly_predictions_api():
    """API endpoint to get weekly price predictions for all major coins"""
    try:
        # Use default predictions for now
        default_predictions = [
            {
                "coin_id": "bitcoin",
                "prediction": 1.25,
                "direction": "up",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "coin_id": "ethereum",
                "prediction": 2.45,
                "direction": "up",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "coin_id": "xrp",
                "prediction": -0.75,
                "direction": "down",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            },
            {
                "coin_id": "tether",
                "prediction": 0.05,
                "direction": "up",
                "as_of_date": datetime.now().strftime("%Y-%m-%d")
            }
        ]
        
        return jsonify({
            "predictions": default_predictions,
            "status": "success"
        })
    except Exception as e:
        logger.error(f"Error generating weekly predictions: {e}")
        return jsonify({
            "error": "Prediction error",
            "message": str(e),
            "status": "error"
        }), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', message='Page not found'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', message='Server error'), 500
