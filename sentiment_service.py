"""
Sentiment Analysis Service for Visionx Ai Beginners Cryptocurrency Dashboard
Uses VADER and TextBlob with enhanced keyword analysis for sentiment analysis
"""
import os
import json
import logging
import re
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from app import db
from models import SentimentRecord, SentimentMention, DataSourceType

# Download NLTK data needed for TextBlob and VADER
try:
    nltk.download('punkt')
    nltk.download('vader_lexicon')
except:
    logging.warning("Failed to download NLTK data. Sentiment analysis may not function properly.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalysisService:
    """Service to analyze sentiment of cryptocurrency mentions"""
    
    def __init__(self):
        """Initialize the service with available APIs"""
        self.twitter_available = False
        self.reddit_available = False
        self.news_api_available = False
        
        # Initialize sentiment analyzers
        try:
            self.vader = SentimentIntensityAnalyzer()
            self._update_vader_lexicon()
            self.vader_available = True
            logging.info("Using enhanced sentiment analysis")
        except Exception as e:
            self.vader_available = False
            logger.warning(f"VADER sentiment analyzer not available: {e}")
        
        # Check if News API key is available
        if os.environ.get('NEWS_API_KEY'):
            self.news_api_available = True
            self.news_api_key = os.environ.get('NEWS_API_KEY')
            logger.info("News API initialized successfully")
        else:
            logger.warning("NEWS_API_KEY environment variable not found")

    def _update_vader_lexicon(self):
        """Update VADER lexicon with cryptocurrency-specific terms"""
        crypto_lexicon = {
            # Positive terms
            'hodl': 2.0,
            'mooning': 3.0,
            'moon': 2.0,
            'bullish': 3.0,
            'bull': 2.0,
            'breakout': 2.5,
            'alt season': 2.0,
            'adoption': 2.0,
            'halving': 1.5,
            'institutional': 1.0,
            'partnership': 1.5,
            'rally': 2.0,
            'support': 1.0,
            'accumulate': 1.0,
            'long': 0.8,
            'undervalued': 1.5,
            
            # Negative terms
            'bearish': -3.0,
            'bear': -2.0,
            'dump': -2.5,
            'dumping': -3.0,
            'correction': -1.5,
            'crash': -3.0,
            'ban': -2.5,
            'banning': -2.5,
            'banned': -2.0,
            'hack': -3.0,
            'hacked': -3.0,
            'ponzi': -3.5,
            'scam': -3.5,
            'rugpull': -4.0,
            'rug pull': -4.0,
            'short': -0.8,
            'resistance': -0.5,
            'overvalued': -1.5,
            'fud': -2.0,
            'fear': -1.5,
            'uncertainty': -1.0,
            'doubt': -1.0,
            'regulation': -0.5,  # Slightly negative by default
            'sec': -0.5,  # Securities and Exchange Commission (context dependent)
            'delisting': -2.5,
            'manipulation': -2.0,
            'bubble': -2.0,
            'sell-off': -2.0,
            'selloff': -2.0,
            'whale': -0.5,  # Slightly negative (large holders can move markets)
        }
        
        # Update the lexicon
        for term, score in crypto_lexicon.items():
            self.vader.lexicon[term] = score
    
    def _preprocess_text(self, text):
        """Clean and prepare text for sentiment analysis"""
        # Convert to lowercase and remove extra whitespace
        text = ' '.join(text.lower().split())
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Replace emoji patterns with words
        emoji_map = {
            r'😊|😄|😁|🙂|😃|😀': ' happy ',
            r'🚀|🌙': ' moon bullish ',
            r'📈': ' up gain ',
            r'📉': ' down loss ',
            r'😢|😭|😞|☹': ' sad ',
            r'😠|😡|🤬': ' angry ',
            r'💰|💵|💲': ' money ',
            r'👍': ' good ',
            r'👎': ' bad ',
            r'❤|♥': ' love ',
            r'💩': ' bad ',
            r'💯': ' perfect ',
            r'🔥': ' hot trending ',
            r'💎': ' diamond hands hodl ',
            r'🐻': ' bear bearish ',
            r'🐂|🐮': ' bull bullish ',
            r'🤔': ' thinking ',
        }
        
        # Replace emoji patterns with corresponding words
        for pattern, replacement in emoji_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Replace crypto slang with standard terms
        slang_map = {
            r'\bhodl\b': 'hold',
            r'\bhv\b': 'halving',
            r'\bwagmi\b': 'we are going to make it',
            r'\bngmi\b': 'not going to make it',
            r'\bdyor\b': 'do your own research',
            r'\bfomo\b': 'fear of missing out',
            r'\bbtfd\b': 'buy the dip',
            r'\bbtd\b': 'buy the dip',
            r'\bgm\b': 'good morning',
            r'\bgs\b': 'good ser',
            r'\botc\b': 'over the counter',
            r'\bliqd\b': 'liquidated',
            r'\bltc\b': 'litecoin',
            r'\bbtc\b': 'bitcoin',
            r'\beth\b': 'ethereum',
            r'\bxrp\b': 'ripple',
            r'\bada\b': 'cardano',
            r'\bsol\b': 'solana',
            r'\bbnb\b': 'binance',
            r'\bdot\b': 'polkadot',
            r'\bmatic\b': 'polygon',
        }
        
        # Replace slang with standard terms
        for pattern, replacement in slang_map.items():
            text = re.sub(pattern, replacement, text)
            
        return text
    
    def analyze_text(self, text, coin_id=None):
        """
        Analyze the sentiment of a text using VADER and TextBlob with enhanced crypto analysis
        
        Args:
            text (str): The text to analyze
            coin_id (str, optional): Cryptocurrency ID for context-specific analysis
            
        Returns:
            dict: Sentiment analysis results with score and magnitude
        """
        if not text or len(text.strip()) < 5:  # Ignore very short texts
            logger.warning("Text too short for sentiment analysis")
            return {"score": 0, "magnitude": 0, "source": "none"}
        
        # Clean and preprocess the text
        text = self._preprocess_text(text)
        
        # First try VADER if available (better for social media)
        if self.vader_available:
            try:
                # Get VADER sentiment scores
                vader_scores = self.vader.polarity_scores(text)
                compound_score = vader_scores['compound']  # -1 to 1 scale
                
                # Apply crypto-specific contextual adjustments
                if coin_id:
                    # Check for coin-specific patterns
                    coin_name = coin_id.lower()
                    
                    # Check for patterns like "buying bitcoin" or "selling ethereum"
                    buy_pattern = re.search(f"(buy|buying|bought|accumulate|long) .{0,20}?{coin_name}", text)
                    sell_pattern = re.search(f"(sell|selling|sold|dump|short) .{0,20}?{coin_name}", text)
                    
                    if buy_pattern:
                        compound_score = min(1.0, compound_score + 0.2)  # Boost positive
                    elif sell_pattern:
                        compound_score = max(-1.0, compound_score - 0.2)  # Boost negative
                
                # Create magnitude (intensity) based on polarity extremes and text length
                word_count = len(text.split())
                magnitude = min(30, max(1.0, abs(compound_score) * word_count * 0.15))
                
                logger.debug(f"VADER sentiment analysis: score={compound_score}, magnitude={magnitude}")
                return {
                    "score": compound_score,
                    "magnitude": magnitude,
                    "positive": vader_scores['pos'],
                    "negative": vader_scores['neg'],
                    "neutral": vader_scores['neu'],
                    "source": "vader"
                }
                
            except Exception as e:
                logger.warning(f"VADER analysis error: {e}. Falling back to TextBlob.")
        
        # Fallback to TextBlob if VADER is not available or fails
        try:
            # Process with TextBlob
            analysis = TextBlob(text)
            
            # TextBlob polarity is between -1 and 1
            # TextBlob subjectivity is between 0 and 1
            polarity = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            # Apply cryptocurrency-specific sentiment adjustment
            crypto_adjustment = self._crypto_keyword_adjustment(text)
            adjusted_polarity = max(-1.0, min(1.0, polarity + crypto_adjustment))
            
            # Create magnitude based on subjectivity and text length
            word_count = len(text.split())
            estimated_magnitude = min(30, max(0.5, subjectivity * word_count * 0.25))
            
            logger.debug(f"TextBlob sentiment analysis: polarity={adjusted_polarity}, magnitude={estimated_magnitude}")
            return {
                "score": adjusted_polarity,
                "magnitude": estimated_magnitude,
                "source": "textblob"
            }
        except Exception as e:
            logger.error(f"TextBlob analysis error: {e}")
            # Fallback to basic keyword matching if TextBlob fails
            try:
                return self._basic_keyword_sentiment(text)
            except:
                return {"score": 0, "magnitude": 0, "source": "fallback"}
                
    def _crypto_keyword_adjustment(self, text):
        """
        Apply cryptocurrency-specific sentiment adjustments based on domain knowledge
        Returns a small adjustment to add to the base sentiment score
        """
        text = text.lower()
        
        # Crypto-specific positive sentiment keywords with weights
        crypto_pos = {
            'adoption': 0.2,
            'institutional': 0.15,
            'launch': 0.1,
            'partnership': 0.15,
            'breakthrough': 0.2,
            'halving': 0.1,
            'all-time high': 0.2,
            'ath': 0.15,
            'moon': 0.05,  # Slang, less weight
            'hodl': 0.05,  # Slang, less weight
            'accumulate': 0.1,
            'bullrun': 0.15,
            'rally': 0.1,
            'upgrade': 0.1,
            'regulation': 0.05,  # Can be positive in context of clarity
        }
        
        # Crypto-specific negative sentiment keywords with weights
        crypto_neg = {
            'hack': -0.2,
            'scam': -0.2,
            'ponzi': -0.2,
            'ban': -0.15,
            'bubble': -0.15,
            'crash': -0.15,
            'dump': -0.1,
            'rugpull': -0.2,
            'delisting': -0.15,
            'liquidation': -0.1,
            'exploit': -0.15,
            'attack': -0.1,
            'fud': -0.05,  # Slang, less weight
            'selloff': -0.1,
            'correction': -0.05,
        }
        
        # Calculate weighted adjustment
        adjustment = 0
        
        # Check for positive crypto terms
        for term, weight in crypto_pos.items():
            if term in text:
                adjustment += weight
        
        # Check for negative crypto terms
        for term, weight in crypto_neg.items():
            if term in text:
                adjustment += weight  # Weight is already negative
        
        # Cap the adjustment to a reasonable range
        return max(-0.3, min(0.3, adjustment))
    
    def _basic_keyword_sentiment(self, text):
        """Very basic sentiment analysis using keyword matching as a last resort"""
        text = text.lower()
        positive_words = ['good', 'great', 'excellent', 'positive', 'bull', 'bullish', 'up', 
                          'rise', 'rising', 'grow', 'growth', 'profit', 'gain', 'success']
        negative_words = ['bad', 'poor', 'negative', 'bear', 'bearish', 'down', 'fall', 'falling',
                          'drop', 'lose', 'loss', 'fail', 'failure', 'crash', 'crisis']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return {"score": 0, "magnitude": 0, "source": "keyword"}
        
        score = (pos_count - neg_count) / total  # Range: -1.0 to 1.0
        magnitude = total * 0.5  # Simple scaling
        
        return {"score": score, "magnitude": magnitude, "source": "keyword"}
    
    def store_sentiment_data(self, coin_id, source_type, sentiment_score, magnitude, 
                            volume=1, content=None, url=None, author=None):
        """
        Store sentiment analysis results in the database
        
        Args:
            coin_id (str): The cryptocurrency ID (e.g., 'bitcoin')
            source_type (DataSourceType): The source of the data
            sentiment_score (float): The sentiment score (-1 to 1)
            magnitude (float): The sentiment magnitude (0 to +inf)
            volume (int): Number of mentions analyzed
            content (str, optional): The original text content
            url (str, optional): URL to the original content
            author (str, optional): Author of the content
        """
        try:
            # Store the aggregate sentiment record
            sentiment_record = SentimentRecord(
                coin_id=coin_id,
                source=source_type,
                sentiment_score=sentiment_score,
                magnitude=magnitude,
                volume=volume,
                created_at=datetime.utcnow()
            )
            db.session.add(sentiment_record)
            
            # Store the individual mention if content is provided
            if content:
                mention = SentimentMention(
                    coin_id=coin_id,
                    source=source_type,
                    content=content,
                    sentiment_score=sentiment_score,
                    url=url,
                    author=author,
                    created_at=datetime.utcnow()
                )
                db.session.add(mention)
            
            db.session.commit()
            return True
        except Exception as e:
            db.session.rollback()
            logger.error(f"Failed to store sentiment data: {e}")
            return False
    
    def get_sentiment_for_coin(self, coin_id, days=7):
        """
        Get sentiment data for a specific coin
        
        Args:
            coin_id (str): The cryptocurrency ID
            days (int): Number of days of data to retrieve
            
        Returns:
            dict: Sentiment data organized by source
        """
        try:
            since_date = datetime.utcnow() - timedelta(days=days)
            
            # Query for aggregated sentiment records
            records = SentimentRecord.query.filter(
                SentimentRecord.coin_id == coin_id,
                SentimentRecord.created_at >= since_date
            ).order_by(SentimentRecord.created_at.asc()).all()
            
            # Query for recent mentions
            mentions = SentimentMention.query.filter(
                SentimentMention.coin_id == coin_id,
                SentimentMention.created_at >= since_date
            ).order_by(SentimentMention.created_at.desc()).limit(50).all()
            
            # Organize data by source
            result = {
                "overall": {
                    "average_score": 0,
                    "total_volume": 0,
                    "timeline": []
                }
            }
            
            # Initialize source-specific data
            for source in DataSourceType:
                result[source.value] = {
                    "average_score": 0,
                    "total_volume": 0,
                    "timeline": []
                }
            
            # Process records
            total_score_weighted = 0
            total_volume = 0
            
            for record in records:
                source = record.source.value
                
                # Add to source timeline
                result[source]["timeline"].append({
                    "timestamp": record.created_at.isoformat(),
                    "score": record.sentiment_score,
                    "magnitude": record.magnitude,
                    "volume": record.volume
                })
                
                # Update source metrics
                result[source]["total_volume"] += record.volume
                
                # Update overall metrics
                total_score_weighted += record.sentiment_score * record.volume
                total_volume += record.volume
            
            # Calculate averages
            for source in DataSourceType:
                source_val = source.value
                if result[source_val]["total_volume"] > 0:
                    source_total = 0
                    for point in result[source_val]["timeline"]:
                        source_total += point["score"] * point["volume"]
                    result[source_val]["average_score"] = source_total / result[source_val]["total_volume"]
            
            # Calculate overall average
            if total_volume > 0:
                result["overall"]["average_score"] = total_score_weighted / total_volume
                result["overall"]["total_volume"] = total_volume
            
            # Format mentions
            result["recent_mentions"] = []
            for mention in mentions:
                result["recent_mentions"].append({
                    "source": mention.source.value,
                    "content": mention.content,
                    "score": mention.sentiment_score,
                    "url": mention.url,
                    "author": mention.author,
                    "timestamp": mention.created_at.isoformat()
                })
            
            return result
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {e}")
            return None

# Initialize the service
sentiment_service = SentimentAnalysisService()