import requests
import logging
import time
from app import cache
from datetime import datetime
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# CoinGecko API base URL
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"

# Cache durations in seconds - significantly increased to handle rate limits
MARKET_DATA_CACHE = 3600     # 1 hour 
COIN_DATA_CACHE = 7200       # 2 hours
HISTORICAL_DATA_CACHE = 28800  # 8 hours
CURRENCY_LIST_CACHE = 86400  # 24 hours - currencies rarely change

# API rate limiting parameters
MAX_RETRIES = 5
RETRY_DELAY = 10  # Base delay in seconds

def handle_rate_limit(func):
    """Decorator to handle API rate limiting with exponential backoff"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        while retries < MAX_RETRIES:
            try:
                return func(*args, **kwargs)
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None and e.response.status_code == 429:
                    # This is a rate limit error
                    retries += 1
                    if retries >= MAX_RETRIES:
                        logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries for {func.__name__}")
                        break
                        
                    # Calculate exponential backoff time
                    wait_time = RETRY_DELAY * (2 ** (retries - 1))
                    logger.warning(f"Rate limit hit, retrying in {wait_time}s ({retries}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    # This is another type of error
                    logger.error(f"Error in {func.__name__}: {e}")
                    break
        
        # Return an appropriate fallback based on the function
        if func.__name__ == 'get_coin_details':
            return None
        elif func.__name__ == 'get_coin_market_chart':
            return None
        elif 'get_top_coins' in func.__name__:
            return []
        elif 'get_global_data' in func.__name__ or 'get_trending_coins' in func.__name__ or 'search_coins' in func.__name__:
            return {}
        else:
            return None
            
    return wrapper

@cache.memoize(timeout=MARKET_DATA_CACHE)
@handle_rate_limit
def get_top_coins(currency='usd', count=50, page=1):
    """Fetch top cryptocurrencies by market cap."""
    url = f"{COINGECKO_API_URL}/coins/markets"
    params = {
        'vs_currency': currency,
        'order': 'market_cap_desc',
        'per_page': count,
        'page': page,
        'sparkline': 'true',
        'price_change_percentage': '1h,24h,7d'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

@cache.memoize(timeout=MARKET_DATA_CACHE)
@handle_rate_limit
def get_global_data():
    """Fetch global cryptocurrency market data."""
    url = f"{COINGECKO_API_URL}/global"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@cache.memoize(timeout=MARKET_DATA_CACHE)
@handle_rate_limit
def get_trending_coins():
    """Fetch trending coins over the last 24 hours."""
    url = f"{COINGECKO_API_URL}/search/trending"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

@cache.memoize(timeout=COIN_DATA_CACHE)
@handle_rate_limit
def get_coin_details(coin_id):
    """Fetch detailed information about a specific coin."""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}"
    params = {
        'localization': 'false',
        'tickers': 'true',
        'market_data': 'true',
        'community_data': 'true',
        'developer_data': 'false',
        'sparkline': 'true'
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

@cache.memoize(timeout=HISTORICAL_DATA_CACHE)
@handle_rate_limit
def get_coin_market_chart(coin_id, currency='usd', days='30'):
    """Fetch historical market data for a specific coin."""
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': currency,
        'days': days,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

@cache.memoize(timeout=MARKET_DATA_CACHE)
@handle_rate_limit
def search_coins(query):
    """Search for cryptocurrencies by name or symbol."""
    url = f"{COINGECKO_API_URL}/search"
    params = {'query': query}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

# Common currencies as fallback if API is unavailable
DEFAULT_CURRENCIES = ["usd", "eur", "gbp", "jpy", "aud", "cad", "chf", "cny", "inr", "btc", "eth"]

@cache.memoize(timeout=CURRENCY_LIST_CACHE)
@handle_rate_limit
def get_supported_currencies():
    """Get list of supported vs_currencies for price conversion."""
    try:
        url = f"{COINGECKO_API_URL}/simple/supported_vs_currencies"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch supported currencies: {e}. Using default list.")
        return DEFAULT_CURRENCIES

def format_market_data(coins):
    """Format market data for display."""
    formatted_coins = []
    for coin in coins:
        formatted_coin = {
            'id': coin.get('id'),
            'symbol': coin.get('symbol', '').upper(),
            'name': coin.get('name'),
            'image': coin.get('image'),
            'current_price': coin.get('current_price'),
            'market_cap': coin.get('market_cap'),
            'market_cap_rank': coin.get('market_cap_rank'),
            'total_volume': coin.get('total_volume'),
            'price_change_24h': coin.get('price_change_24h'),
            'price_change_percentage_24h': coin.get('price_change_percentage_24h'),
            'price_change_percentage_1h_in_currency': coin.get('price_change_percentage_1h_in_currency'),
            'price_change_percentage_7d_in_currency': coin.get('price_change_percentage_7d_in_currency'),
            'sparkline_in_7d': coin.get('sparkline_in_7d', {}).get('price', []),
            'last_updated': coin.get('last_updated')
        }
        formatted_coins.append(formatted_coin)
    return formatted_coins

def format_timestamp_data(chart_data):
    """Format timestamp data for Chart.js."""
    if not chart_data:
        return None
    
    prices = chart_data.get('prices', [])
    formatted_data = {
        'labels': [],
        'prices': []
    }
    
    for price_data in prices:
        timestamp, price = price_data
        # Convert timestamp (milliseconds) to readable date
        date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M')
        formatted_data['labels'].append(date)
        formatted_data['prices'].append(price)
    
    return formatted_data
