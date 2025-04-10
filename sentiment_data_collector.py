"""
Sentiment Data Collector for Visionx Ai Beginners Cryptocurrency Dashboard
Collects and analyzes sentiment data from various sources.
"""
import os
import time
import json
import logging
import argparse
import requests
from datetime import datetime, timedelta
import random  # For demo purposes only, to simulate data

# Import Flask app and models in application context
from app import app, db
from models import SentimentRecord, SentimentMention, DataSourceType
# Import improved sentiment analysis
try:
    from enhanced_sentiment import analyze_sentiment, batch_analyze, store_sentiment_data
    from sentiment_service import sentiment_service  # Import this anyway for fallback
    USING_ENHANCED = True
    logging.info("Using enhanced sentiment analysis")
except ImportError:
    from sentiment_service import sentiment_service
    USING_ENHANCED = False
    logging.info("Using standard sentiment analysis")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentDataCollector:
    """Collects sentiment data from various sources and stores in the database."""
    
    def __init__(self):
        """Initialize the collector with API credentials."""
        # Check for environment variables with API credentials
        self.twitter_available = bool(os.environ.get('TWITTER_API_KEY') and 
                                       os.environ.get('TWITTER_API_SECRET'))
        self.reddit_available = bool(os.environ.get('REDDIT_CLIENT_ID') and 
                                      os.environ.get('REDDIT_CLIENT_SECRET'))
        self.news_available = bool(os.environ.get('NEWS_API_KEY'))
        
        # Initialize services if credentials are available
        if self.twitter_available:
            logger.info("Twitter API credentials found. Initializing Twitter client.")
            # Initialize Twitter client here
        else:
            logger.warning("Twitter API credentials not found in environment variables.")
        
        if self.reddit_available:
            logger.info("Reddit API credentials found. Initializing Reddit client.")
            # Initialize Reddit client here
        else:
            logger.warning("Reddit API credentials not found in environment variables.")
        
        if self.news_available:
            self.news_api_key = os.environ.get('NEWS_API_KEY')
            logger.info("News API credentials found. Initializing News API client.")
        else:
            logger.warning("NEWS_API_KEY not found in environment variables.")
    
    def collect_twitter_sentiment(self, coin_id, query=None, count=100):
        """
        Twitter sentiment is not implemented - this is a placeholder for UI compatibility.
        The UI shows Twitter data sections, but we don't actually collect Twitter data.
        
        Args:
            coin_id (str): The cryptocurrency ID (e.g., 'bitcoin')
            query (str, optional): Custom search query. If None, uses the coin_id.
            count (int): Number of tweets to analyze.
            
        Returns:
            dict: Empty result as Twitter API is not used
        """
        logger.info(f"Twitter data collection is not enabled for {coin_id}")
        return None
    
    def collect_reddit_sentiment(self, coin_id, subreddits=None, time_filter='day', limit=100):
        """
        Reddit sentiment is not implemented - this is a placeholder for UI compatibility.
        The UI shows Reddit data sections, but we don't actually collect Reddit data.
        
        Args:
            coin_id (str): The cryptocurrency ID (e.g., 'bitcoin')
            subreddits (list, optional): List of subreddits to search. If None, uses defaults.
            time_filter (str): Time filter for posts ('hour', 'day', 'week', 'month', 'year', 'all')
            limit (int): Maximum number of posts to analyze.
            
        Returns:
            dict: Empty result as Reddit API is not used
        """
        logger.info(f"Reddit data collection is not enabled for {coin_id}")
        return None
    
    def collect_news_sentiment(self, coin_id, days=1, language='en'):
        """
        Collect and analyze news sentiment for a cryptocurrency.
        
        Args:
            coin_id (str): The cryptocurrency ID (e.g., 'bitcoin')
            days (int): Number of days of news to analyze.
            language (str): Language of news articles.
            
        Returns:
            dict: Sentiment analysis results with scores and mentions
        """
        if not self.news_available:
            logger.warning(f"News API not available. Skipping news sentiment for {coin_id}")
            return None
        
        try:
            logger.info(f"Collecting news sentiment for {coin_id} from the past {days} days")
            
            # Get API key from environment variable
            api_key = os.environ.get('NEWS_API_KEY')
            if api_key:
                logger.info(f"Using News API key: {api_key[:5]}...{api_key[-3:] if len(api_key) > 8 else ''}")
            else:
                logger.error("NEWS_API_KEY is not set in environment variables")
            
            # Calculate date range
            from_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
            logger.info(f"Searching for articles since: {from_date}")
            
            # Query parameters - more specific for better results
            # For Bitcoin, we use a more targeted query to ensure relevant articles
            if coin_id.lower() == 'bitcoin':
                query = f'"{coin_id}" AND (crypto OR cryptocurrency OR blockchain OR price OR market)'
            else:
                query = f'"{coin_id}" AND (crypto OR cryptocurrency)'
            
            logger.info(f"Using search query: {query}")
                
            params = {
                'q': query,
                'sortBy': 'relevancy',
                'language': language,
                'from': from_date,
                'pageSize': 20,  # Reduced to avoid rate limits
                'apiKey': api_key
            }
            
            # Make API request
            url = 'https://newsapi.org/v2/everything'
            logger.info(f"Making request to News API: {url} with params (excluding apiKey): {str({k:v for k,v in params.items() if k != 'apiKey'})}")
            response = requests.get(url, params=params)
            
            logger.info(f"News API response status code: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            
            # Process results
            articles = data.get('articles', [])
            logger.info(f"Found {len(articles)} news articles for {coin_id}")
            
            if not articles:
                logger.warning("No articles found for the given search criteria")
                return None
                
            # Analyze sentiment for each article
            total_score = 0
            total_magnitude = 0
            mentions = []
            
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                url = article.get('url', '')
                source = article.get('source', {}).get('name', 'Unknown')
                
                # Combine available text for analysis
                text = f"{title} {description} {content}"
                if not text.strip():
                    continue
                
                # Analyze sentiment using either enhanced or standard analyzer
                if USING_ENHANCED:
                    # Use enhanced sentiment analysis with coin context
                    sentiment = analyze_sentiment(text, coin=coin_id, source='news')
                    score = sentiment.get('score', 0)
                    magnitude = sentiment.get('magnitude', 0)
                else:
                    # Use standard sentiment analysis
                    sentiment = sentiment_service.analyze_text(text)
                    score = sentiment.get('score', 0)
                    magnitude = sentiment.get('magnitude', 0)
                
                logger.info(f"Sentiment for article '{title[:30]}...': score={score:.2f}, magnitude={magnitude:.2f}")
                
                total_score += score
                total_magnitude += magnitude
                
                # Store as a mention if it has significant sentiment
                if abs(score) > 0.2 or magnitude > 0.8:  # Lower threshold to capture more mentions
                    mentions.append({
                        'content': title,
                        'score': score,
                        'url': url,
                        'author': source
                    })
                    
                    # Store in the database using either enhanced or standard method
                    if USING_ENHANCED:
                        store_sentiment_data(
                            coin_id=coin_id,
                            source_type=DataSourceType.NEWS,
                            sentiment_score=score,
                            magnitude=magnitude,
                            content=title,
                            url=url,
                            author=source
                        )
                    else:
                        sentiment_service.store_sentiment_data(
                            coin_id=coin_id,
                            source_type=DataSourceType.NEWS,
                            sentiment_score=score,
                            magnitude=magnitude,
                            content=title,
                            url=url,
                            author=source
                        )
            
            # Calculate average sentiment
            if articles:
                avg_score = total_score / len(articles)
                avg_magnitude = total_magnitude / len(articles)
                
                # Store aggregate sentiment using either enhanced or standard method
                if USING_ENHANCED:
                    store_sentiment_data(
                        coin_id=coin_id,
                        source_type=DataSourceType.NEWS,
                        sentiment_score=avg_score,
                        magnitude=avg_magnitude,
                        volume=len(articles)
                    )
                else:
                    sentiment_service.store_sentiment_data(
                        coin_id=coin_id,
                        source_type=DataSourceType.NEWS,
                        sentiment_score=avg_score,
                        magnitude=avg_magnitude,
                        volume=len(articles)
                    )
                
                return {
                    'score': avg_score,
                    'magnitude': avg_magnitude,
                    'volume': len(articles),
                    'mentions': mentions
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error collecting news sentiment: {e}")
            return None
    
    def create_sample_data(self, coin_ids, days=7):
        """
        Create sample sentiment data for demonstration purposes.
        This is for demonstration only and should be replaced with real data.
        
        Args:
            coin_ids (list): List of cryptocurrency IDs
            days (int): Number of days of data to generate
        """
        logger.info(f"Creating sample sentiment data for {len(coin_ids)} coins over {days} days")
        
        with app.app_context():
            start_date = datetime.utcnow() - timedelta(days=days)
            
            for coin_id in coin_ids:
                # Create sentiment records for each source and day
                for source in [DataSourceType.TWITTER, DataSourceType.REDDIT, DataSourceType.NEWS]:
                    for day in range(days):
                        current_date = start_date + timedelta(days=day)
                        
                        # Create a few records per day to simulate time series
                        for hour in [10, 14, 18, 22]:
                            timestamp = current_date.replace(hour=hour)
                            
                            # Generate random sentiment score between -1 and 1
                            # With some consistency per coin and source
                            base_score = random.uniform(-0.5, 0.5)
                            coin_factor = hash(coin_id) % 100 / 100.0 * 0.5
                            source_factor = hash(source.value) % 100 / 100.0 * 0.5
                            daily_factor = day / days * 0.2
                            
                            sentiment_score = max(-1.0, min(1.0, base_score + coin_factor + source_factor + daily_factor))
                            magnitude = random.uniform(0.5, 2.0)
                            volume = random.randint(5, 50)
                            
                            # Create and save the record
                            record = SentimentRecord(
                                coin_id=coin_id,
                                source=source,
                                sentiment_score=sentiment_score,
                                magnitude=magnitude,
                                volume=volume,
                                created_at=timestamp
                            )
                            db.session.add(record)
                        
                        # Create sample mentions (5-10 per day per source)
                        mention_count = random.randint(5, 10)
                        for i in range(mention_count):
                            # Random time during the day
                            hour = random.randint(0, 23)
                            minute = random.randint(0, 59)
                            mention_time = current_date.replace(hour=hour, minute=minute)
                            
                            # Generate random sentiment score between -1 and 1
                            sentiment_score = random.uniform(-1.0, 1.0)
                            
                            # Sample content templates
                            templates = {
                                DataSourceType.TWITTER: [
                                    f"Just bought some #{coin_id.upper()}! Let's go to the moon! 🚀 #crypto #investment",
                                    f"Not feeling good about {coin_id.upper()} right now. The market looks bearish. #crypto",
                                    f"Is {coin_id.capitalize()} a good investment right now? What do you think? #cryptocurrency",
                                    f"{coin_id.capitalize()} is showing strong fundamentals despite market conditions. #HODL",
                                    f"Why is {coin_id.upper()} dropping today? Any news I missed? #crypto #markets"
                                ],
                                DataSourceType.REDDIT: [
                                    f"Analysis: Why {coin_id.capitalize()} might be undervalued right now",
                                    f"Technical Analysis of {coin_id.upper()} - Bearish pattern forming?",
                                    f"Long-term holder of {coin_id.capitalize()} here - my thoughts on recent developments",
                                    f"Is anyone else concerned about the recent {coin_id.capitalize()} price action?",
                                    f"Discussion: {coin_id.capitalize()}'s technology advantages over competitors"
                                ],
                                DataSourceType.NEWS: [
                                    f"{coin_id.capitalize()} Price Jumps 10% Following Positive Regulatory News",
                                    f"Analysts: {coin_id.upper()} Could Reach New Highs By End Of Year",
                                    f"Major Companies Begin Accepting {coin_id.capitalize()} as Payment",
                                    f"Concerns Grow Over {coin_id.capitalize()}'s Energy Consumption",
                                    f"New {coin_id.capitalize()} Update Promises Improved Scalability"
                                ]
                            }
                            
                            # Select a template based on sentiment direction
                            available_templates = templates[source]
                            if sentiment_score > 0.3:
                                # Positive sentiment - use positive templates (0, 3)
                                template_idx = random.choice([0, 3])
                            elif sentiment_score < -0.3:
                                # Negative sentiment - use negative templates (1, 4)
                                template_idx = random.choice([1, 4])
                            else:
                                # Neutral sentiment - use neutral template (2)
                                template_idx = 2
                                
                            content = available_templates[template_idx % len(available_templates)]
                            
                            authors = {
                                DataSourceType.TWITTER: [
                                    "CryptoTrader", "BlockchainBob", "SatoshiFan", "CoinAnalyst", "CryptoQueen"
                                ],
                                DataSourceType.REDDIT: [
                                    "crypto_enthusiast", "HODLer2023", "blockchain_dev", "to_the_moon", "long_term_investor"
                                ],
                                DataSourceType.NEWS: [
                                    "CoinDesk", "CryptoNews", "BlockchainTimes", "DigitalAssetReport", "CryptoDaily"
                                ]
                            }
                            
                            author = random.choice(authors[source])
                            
                            # URL generation (fake)
                            base_urls = {
                                DataSourceType.TWITTER: "https://twitter.com/",
                                DataSourceType.REDDIT: "https://reddit.com/r/",
                                DataSourceType.NEWS: "https://crypto-news-example.com/"
                            }
                            url = f"{base_urls[source]}{coin_id.lower()}-{day}-{i}"
                            
                            # Create and save the mention
                            mention = SentimentMention(
                                coin_id=coin_id,
                                source=source,
                                content=content,
                                sentiment_score=sentiment_score,
                                author=author,
                                url=url,
                                created_at=mention_time
                            )
                            db.session.add(mention)
                
                logger.info(f"Created sample data for {coin_id}")
            
            # Commit all changes
            db.session.commit()
            logger.info("Sample data creation complete")

    def collect_and_analyze(self, coin_ids, sources=None, use_sample_data=False):
        """
        Collect and analyze sentiment for a list of cryptocurrencies.
        This implementation only uses NewsAPI for authentic data while keeping 
        the UI elements for Twitter and Reddit.
        
        Args:
            coin_ids (list): List of cryptocurrency IDs
            sources (list, optional): Which sources to use ['news', 'twitter', 'reddit']. Only 'news' will actually collect data.
            use_sample_data (bool): Whether to generate sample data instead of collecting real data
        """
        with app.app_context():
            if use_sample_data:
                logger.warning("Sample data generation is disabled. Using real news data instead.")
                # We don't generate sample data for data integrity reasons
            
            logger.info(f"Starting sentiment collection for {len(coin_ids)} coins")
            
            # We only collect data from NewsAPI
            if not self.news_available:
                logger.error("NewsAPI key is not available. Cannot collect sentiment data.")
                return
            
            # Log which sources we're using (only news is actually used)
            logger.info("Active sentiment source: news (NewsAPI)")
            
            for coin_id in coin_ids:
                try:
                    logger.info(f"Processing sentiment for {coin_id}")
                    data_collected = False
                    
                    # Collect news sentiment data
                    logger.info(f"Collecting news sentiment for {coin_id}")
                    news_data = self.collect_news_sentiment(coin_id, days=7)
                    if news_data:
                        logger.info(f"Successfully collected news sentiment for {coin_id}")
                        data_collected = True
                    else:
                        logger.warning(f"No news sentiment data available for {coin_id}")
                    
                    # Log that Twitter and Reddit are not being used
                    logger.info(f"Twitter and Reddit data collection is disabled.")
                    
                    if not data_collected:
                        logger.warning(f"No sentiment data available for {coin_id}")
                    else:
                        logger.info(f"Completed sentiment data collection for {coin_id}")
                    
                    # Sleep to avoid overwhelming APIs
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error processing sentiment for {coin_id}: {e}", exc_info=True)
        
        logger.info("Sentiment collection complete")

def main():
    """Main entry point for the sentiment data collector script."""
    parser = argparse.ArgumentParser(description='Collect cryptocurrency sentiment data')
    parser.add_argument('--coins', '-c', nargs='+', default=['bitcoin'],
                        help='List of coin IDs to collect sentiment for')
    parser.add_argument('--sample', '-s', action='store_true',
                        help='Generate sample data instead of collecting real data')
    parser.add_argument('--days', '-d', type=int, default=7,
                        help='Number of days of data to collect or generate')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force collection even if recent data exists')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with verbose logging')
    
    args = parser.parse_args()
    
    # Set up more verbose logging if debug is enabled
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('urllib3').setLevel(logging.INFO)
        logging.getLogger('requests').setLevel(logging.INFO)
        logging.debug("Debug logging enabled")
    
    collector = SentimentDataCollector()
    
    if args.sample:
        logger.warning("Sample data is being generated for demonstration purposes only")
        with app.app_context():
            collector.create_sample_data(args.coins, days=args.days)
    elif args.force:
        logger.info(f"Force mode enabled - directly collecting sentiment data")
        with app.app_context():
            for coin_id in args.coins:
                try:
                    logger.info(f"Forcing collection of sentiment data for {coin_id}")
                    
                    # Directly collect news sentiment
                    if collector.news_available:
                        news_data = collector.collect_news_sentiment(coin_id, days=args.days)
                        if news_data:
                            logger.info(f"Successfully collected news sentiment for {coin_id}: {news_data['volume']} articles")
                        else:
                            logger.warning(f"No news sentiment data found for {coin_id}")
                    
                    # Check if data was collected
                    record_count = db.session.query(SentimentRecord).filter(
                        SentimentRecord.coin_id == coin_id,
                        SentimentRecord.created_at >= datetime.utcnow() - timedelta(days=args.days)
                    ).count()
                    
                    mention_count = db.session.query(SentimentMention).filter(
                        SentimentMention.coin_id == coin_id,
                        SentimentMention.created_at >= datetime.utcnow() - timedelta(days=args.days)
                    ).count()
                    
                    logger.info(f"Database now has {record_count} records and {mention_count} mentions for {coin_id}")
                    
                except Exception as e:
                    logger.error(f"Error collecting sentiment data for {coin_id}: {e}", exc_info=True)
    else:
        collector.collect_and_analyze(args.coins)

if __name__ == "__main__":
    main()