"""
Sentiment Analysis Controller for Visionx Ai Beginners Cryptocurrency Dashboard
Coordinates between enhanced sentiment analysis and the standard sentiment service
"""
import logging
import os
from models import DataSourceType
from sentiment_service import sentiment_service

# Configure logging
logger = logging.getLogger(__name__)

# Try to import enhanced sentiment if available
try:
    from enhanced_sentiment import analyze_sentiment, batch_analyze
    USING_ENHANCED_SENTIMENT = True
    logger.info("Enhanced sentiment analysis is available")
except ImportError:
    USING_ENHANCED_SENTIMENT = False
    logger.info("Enhanced sentiment analysis not available, using standard analysis")

def get_sentiment_data(coin_id, days=7, force_refresh=False):
    """
    Get sentiment data for a specific coin
    
    Args:
        coin_id (str): Cryptocurrency ID
        days (int): Number of days of data to retrieve
        force_refresh (bool): Whether to force fetching new data
        
    Returns:
        dict: Complete sentiment data with enhanced analysis
    """
    # First try to get existing data unless force refresh
    if not force_refresh:
        sentiment_data = sentiment_service.get_sentiment_for_coin(coin_id, days)
        if sentiment_data and sentiment_data.get('recent_mentions'):
            logger.info(f"Found existing sentiment data for {coin_id}")
            return enhance_sentiment_data(sentiment_data, coin_id)
    
    # If we need to refresh or don't have data, check for API key
    if os.environ.get('NEWS_API_KEY'):
        logger.info(f"Collecting fresh sentiment data for {coin_id}")
        
        # Import here to avoid circular imports
        from sentiment_data_collector import SentimentDataCollector
        
        # Collect fresh sentiment data
        collector = SentimentDataCollector()
        news_data = collector.collect_news_sentiment(coin_id, days=days)
        
        if news_data:
            logger.info(f"Successfully collected news sentiment for {coin_id}: {news_data.get('volume', 0)} articles")
            
            # Get the newly collected data
            sentiment_data = sentiment_service.get_sentiment_for_coin(coin_id, days)
            
            if sentiment_data:
                logger.info(f"Retrieved newly collected sentiment data for {coin_id}")
                return enhance_sentiment_data(sentiment_data, coin_id)
    else:
        logger.error("NEWS_API_KEY not found in environment")
    
    return None

def enhance_sentiment_data(sentiment_data, coin_id):
    """
    Apply enhanced sentiment analysis to sentiment data if available
    
    Args:
        sentiment_data (dict): Original sentiment data
        coin_id (str): Cryptocurrency ID
        
    Returns:
        dict: Enhanced sentiment data
    """
    # Return original data if enhanced sentiment is not available
    if not USING_ENHANCED_SENTIMENT:
        return sentiment_data
    
    # Add placeholders for Twitter and Reddit to maintain UI compatibility
    if 'twitter' in sentiment_data and (not sentiment_data['twitter']['timeline'] 
                                     or len(sentiment_data['twitter']['timeline']) == 0):
        sentiment_data['twitter'] = {
            "average_score": 0,
            "total_volume": 0,
            "timeline": [],
            "data_available": False  # Flag to indicate this is a placeholder
        }
        
    if 'reddit' in sentiment_data and (not sentiment_data['reddit']['timeline'] 
                                    or len(sentiment_data['reddit']['timeline']) == 0):
        sentiment_data['reddit'] = {
            "average_score": 0,
            "total_volume": 0,
            "timeline": [],
            "data_available": False  # Flag to indicate this is a placeholder
        }
    
    # Process mentions with enhanced sentiment
    if sentiment_data.get('recent_mentions'):
        mentions = sentiment_data['recent_mentions']
        enhanced_mentions = []
        
        # Process each mention with enhanced sentiment analysis
        for mention in mentions:
            if 'content' in mention and mention.get('content'):
                try:
                    # Analyze with enhanced sentiment analyzer
                    enhanced_result = analyze_sentiment(
                        text=mention['content'], 
                        coin=coin_id, 
                        source=mention.get('source', 'news')
                    )
                    
                    # Create enhanced mention with original data plus enhanced analysis
                    enhanced_mention = mention.copy()
                    enhanced_mention['score'] = enhanced_result['score']
                    enhanced_mention['magnitude'] = enhanced_result.get('magnitude', 0)
                    enhanced_mention['positive'] = enhanced_result.get('positive', 0)
                    enhanced_mention['negative'] = enhanced_result.get('negative', 0)
                    enhanced_mention['neutral'] = enhanced_result.get('neutral', 0)
                    enhanced_mention['category'] = enhanced_result.get('category', 'neutral')
                    enhanced_mentions.append(enhanced_mention)
                except Exception as e:
                    # If enhancement fails, keep original mention
                    logger.warning(f"Error enhancing sentiment for mention: {e}")
                    enhanced_mentions.append(mention)
            else:
                # If no content, keep original mention
                enhanced_mentions.append(mention)
        
        # Replace original mentions with enhanced ones
        sentiment_data['recent_mentions'] = enhanced_mentions
        
        # Calculate enhanced overall sentiment
        if enhanced_mentions:
            total_score = sum(m['score'] for m in enhanced_mentions)
            total_magnitude = sum(m.get('magnitude', 0) for m in enhanced_mentions)
            total_positive = sum(m.get('positive', 0) for m in enhanced_mentions)
            total_negative = sum(m.get('negative', 0) for m in enhanced_mentions)
            total_neutral = sum(m.get('neutral', 0) for m in enhanced_mentions)
            count = len(enhanced_mentions)
            
            avg_score = total_score / count
            avg_magnitude = total_magnitude / count
            avg_positive = total_positive / count
            avg_negative = total_negative / count
            avg_neutral = total_neutral / count
            
            # Determine sentiment category
            if avg_score > 0.2:
                category = "very positive"
            elif avg_score > 0.05:
                category = "positive"
            elif avg_score < -0.2:
                category = "very negative"
            elif avg_score < -0.05:
                category = "negative"
            else:
                category = "neutral"
            
            # Update overall sentiment
            sentiment_data['overall'] = {
                "average_score": avg_score,
                "magnitude": avg_magnitude,
                "positive": avg_positive,
                "negative": avg_negative,
                "neutral": avg_neutral,
                "category": category,
                "total_volume": count
            }
            
            logger.info(f"Enhanced sentiment for {coin_id}: {category} ({avg_score:.2f})")
    
    # Collect timeline data
    timeline_data = []
    for source in ['news', 'twitter', 'reddit']:
        if source in sentiment_data and sentiment_data.get(source, {}).get('timeline') and len(sentiment_data[source]['timeline']) > 0:
            timeline_data.extend(sentiment_data[source]['timeline'])
    
    # Format for frontend consumption
    response_data = {
        "status": "ok",
        "sentiment": sentiment_data.get('overall', {}),
        "mentions": sentiment_data.get('recent_mentions', []),
        "timeline": timeline_data,
        "sources": {
            "news": sentiment_data.get('news', {"data_available": False}),
            "twitter": sentiment_data.get('twitter', {"data_available": False}),
            "reddit": sentiment_data.get('reddit', {"data_available": False})
        }
    }
    
    return response_data

def batch_analyze_coins(coin_ids, text_samples):
    """
    Perform batch sentiment analysis on multiple coins
    
    Args:
        coin_ids (list): List of coin IDs
        text_samples (dict): Dictionary of text samples to analyze keyed by coin_id
        
    Returns:
        dict: Results of batch analysis
    """
    results = {}
    
    if USING_ENHANCED_SENTIMENT:
        # Use enhanced batch analysis
        for coin_id in coin_ids:
            if coin_id in text_samples and text_samples[coin_id]:
                results[coin_id] = batch_analyze(text_samples[coin_id], coin=coin_id)
            else:
                # Try to get existing sentiment data
                sentiment_data = sentiment_service.get_sentiment_for_coin(coin_id, days=1)
                if sentiment_data and sentiment_data.get('overall'):
                    results[coin_id] = sentiment_data.get('overall')
                else:
                    results[coin_id] = {"score": 0, "magnitude": 0, "category": "unknown"}
    else:
        # Use standard sentiment analysis
        for coin_id in coin_ids:
            sentiment_data = sentiment_service.get_sentiment_for_coin(coin_id, days=1)
            if sentiment_data and sentiment_data.get('overall'):
                results[coin_id] = sentiment_data.get('overall')
            else:
                results[coin_id] = {"score": 0, "magnitude": 0, "category": "unknown"}
    
    return results