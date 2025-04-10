"""
Enhanced Sentiment Analysis for Visionx Ai Beginners Cryptocurrency Dashboard
Using advanced NLP techniques for better cryptocurrency sentiment analysis
"""
import os
import logging
import re

# Try to import NLP libraries
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logging.warning("NLTK or TextBlob not available. Installing required packages...")
    # Attempt to install the required packages
    try:
        import subprocess
        subprocess.check_call(['pip', 'install', 'nltk', 'textblob'])
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        from textblob import TextBlob
        NLTK_AVAILABLE = True
    except Exception as e:
        logging.error(f"Failed to install NLP packages: {e}")
        NLTK_AVAILABLE = False

# Import database models
from models import DataSourceType
from app import db
from models import SentimentRecord, SentimentMention

# Download NLTK resources if not already available
if NLTK_AVAILABLE:
    try:
        nltk.data.find('vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
        nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analysis for cryptocurrency content"""
    
    def __init__(self):
        """Initialize the enhanced sentiment analyzer with specialized dictionaries"""
        if NLTK_AVAILABLE:
            self.sia = SentimentIntensityAnalyzer()
            self._update_vader_lexicon()
        else:
            self.sia = None
            logger.warning("NLTK is not available. Sentiment analysis will be limited.")
        
        # Initialize crypto-specific contextual analyzers with greatly enhanced Bitcoin-specific terms
        self.crypto_pos_terms = {
            # General crypto positive terms with increased weights
            'adoption': 0.3,
            'institutional': 0.25,
            'launch': 0.2,
            'partnership': 0.2,
            'breakthrough': 0.3,
            'halving': 0.35,  # Enhanced for Bitcoin
            'all-time high': 0.35,
            'ath': 0.35,
            'moon': 0.15,  # Slang, increased weight
            'hodl': 0.2,  # Slang, increased weight
            'accumulate': 0.25,
            'bullrun': 0.3,
            'rally': 0.25,
            'upgrade': 0.2,
            'regulation clarity': 0.2,  # Specific positive regulatory context
            'regulatory framework': 0.2,
            
            # Bitcoin-specific positive terms
            'btc': 0.1,  # Base boost for Bitcoin mentions
            'bitcoin': 0.1,  # Base boost for Bitcoin mentions
            'satoshi': 0.2,
            'sats': 0.15,
            'lightning network': 0.3,
            'ln': 0.2,
            'digital gold': 0.35,
            'store of value': 0.3,
            'hard money': 0.25,
            'etf approval': 0.4,
            'spot etf': 0.35,
            'legal tender': 0.3,
            'el salvador': 0.2,
            'microstrategy': 0.25,
            'saylor': 0.25,
            'scarcity': 0.25,
            'institutional buying': 0.4,
            'institutional investment': 0.4,
            'mass adoption': 0.35,
            'whales buying': 0.3,
            'hash rate increase': 0.25,
            'hash rate high': 0.25,
            'difficulty increase': 0.2,
            'network growth': 0.25,
            'taproot': 0.2,
            'segwit': 0.2,
            'mining profitability': 0.25,
            'miner accumulation': 0.3,
            'fear and greed index': 0.1,  # Neutral base term
            'greed': 0.25,
            'extreme greed': 0.35,
            'top performing': 0.3,
            'investment vehicle': 0.25,
            'inflation hedge': 0.3,
            'anti-inflation': 0.3,
            'deflationary': 0.25,
            'price target': 0.2,
            'price prediction': 0.1,  # Neutral base term
            'bullish divergence': 0.3,
            'bullish pattern': 0.3,
            'dip opportunity': 0.25,
            'buying opportunity': 0.3,
            'support level': 0.2,
            'holding support': 0.25,
            'breaking resistance': 0.3,
            'new all-time high': 0.4,
            'record high': 0.4,
            'trillion market cap': 0.35,
            'milestone': 0.25,
            'adoption curve': 0.2,
            's2f': 0.2,  # Stock to flow model
            'stock to flow': 0.2,
            'on-chain metrics': 0.1,  # Neutral base term
            'positive on-chain': 0.3,
            'strong fundamentals': 0.35,
            'future outlook': 0.1,  # Neutral base term
            'bright future': 0.3,
        }
        
        self.crypto_neg_terms = {
            # General crypto negative terms with increased weights
            'hack': -0.35,
            'scam': -0.35,
            'ponzi': -0.35,
            'ban': -0.3,
            'bubble': -0.3,
            'crash': -0.35,
            'dump': -0.25,
            'rugpull': -0.4,
            'delisting': -0.3,
            'liquidation': -0.25,
            'exploit': -0.35,
            'attack': -0.3,
            'fud': -0.15,  # Slang, increased weight
            'selloff': -0.25,
            'correction': -0.15,
            
            # Bitcoin-specific negative terms
            '51% attack': -0.4,
            'double spend': -0.45,
            'quantum computing threat': -0.35,
            'china ban': -0.35,
            'mining ban': -0.3,
            'regulatory crackdown': -0.4,
            'sec rejection': -0.4,
            'etf rejection': -0.4,
            'security vulnerability': -0.4,
            'scaling issues': -0.3,
            'high transaction fees': -0.25,
            'mempool congestion': -0.2,
            'network congestion': -0.25,
            'energy consumption': -0.25,
            'environmental concerns': -0.3,
            'esg concerns': -0.25,
            'carbon footprint': -0.25,
            'government regulation': -0.25,
            'regulatory uncertainty': -0.3,
            'exchange hack': -0.4,
            'stolen bitcoin': -0.4,
            'whale selling': -0.35,
            'miner capitulation': -0.35,
            'hash rate decline': -0.3,
            'difficulty drop': -0.25,
            'market manipulation': -0.35,
            'wash trading': -0.3,
            'bearish divergence': -0.3,
            'bearish pattern': -0.3,
            'death cross': -0.35,
            'breaking support': -0.3,
            'key support broken': -0.35,
            'resistance level': -0.2,
            'strong resistance': -0.25,
            'stuck below': -0.25,
            'downtrend': -0.3,
            'bear market': -0.35,
            'fear': -0.25,
            'extreme fear': -0.35,
            'fear and greed index low': -0.3,
            'worst performing': -0.3,
            'losing value': -0.3,
            'price drop': -0.3,
            'massive losses': -0.35,
            'free fall': -0.4,
            'plunge': -0.35,
            'nosedive': -0.35,
            'negative on-chain': -0.3,
            'weak fundamentals': -0.35,
            'tether concerns': -0.3,
            'stablecoin risk': -0.25,
            'liquidity crisis': -0.35,
            'exodus': -0.3,
            'mass selling': -0.35,
            'capitulation': -0.35,
            'panic selling': -0.4,
            'tax evasion': -0.35,
            'money laundering': -0.35,
            'illicit activities': -0.35,
            'terrorist financing': -0.45,
            'criminal use': -0.4,
            'dark web': -0.3,
            'futures liquidation': -0.35,
            'margin call': -0.3,
            'over-leveraged': -0.25,
            'bearish outlook': -0.3,
            'negative sentiment': -0.25,
        }
        
        # Enhanced emoji mappings with stronger Bitcoin sentiment
        self.emoji_map = {
            # Positive sentiment emojis
            r'😊|😄|😁|🙂|😃|😀': ' happy very positive ',
            r'🚀': ' moon rocket bullish very positive bitcoin surge ',  # Rocket - strongly bullish
            r'🌙': ' moon target bullish positive bitcoin ',  # Moon - price target
            r'📈': ' up gain increase profit bullish positive ',  # Chart up
            r'💹': ' market up gain profit bullish positive ',  # Chart with upward trend and yen
            r'👑': ' king leader top dominant bitcoin ',  # Crown - Bitcoin as king
            r'🏆': ' winner success achievement best bitcoin ',  # Trophy - success
            r'📊': ' chart analysis data metrics bullish ',  # Bar chart - analysis
            r'🔝': ' top highest peak new record bitcoin ',  # Top arrow - new highs
            r'📣': ' announcement news alert important bitcoin ',  # Megaphone - announcement
            r'💯': ' perfect hundred percent complete bullish ',  # 100 - perfect
            r'🔥': ' hot trending fire popular bullish bitcoin ',  # Fire - trending/hot
            r'💎': ' diamond hands hodl holding strong long-term ',  # Diamond - hold strategy
            r'✅': ' confirmed verify approved positive bitcoin ',  # Check mark - confirmation
            r'🎯': ' target goal achieved successful bitcoin ',  # Target - goal achieved
            r'🚁': ' helicopter money wealth profit bitcoin ',  # Helicopter - money printing/inflation
            r'⚡': ' lightning fast network bitcoin scaling ',  # Lightning - Bitcoin Lightning Network
            r'🏦': ' bank institution financial adoption bitcoin ',  # Bank - institutional adoption
            r'⛓️': ' blockchain technology secure bitcoin ',  # Chain - blockchain
            r'🔒': ' secure locked safe trustworthy bitcoin ',  # Lock - security
            r'💪': ' strong powerful stable resilient bitcoin ',  # Flexed biceps - strength
            r'🤑': ' money rich profit wealth bullish bitcoin ',  # Money mouth face - profit
            r'🐂|🐮': ' bull bullish market uptrend positive bitcoin ',  # Bull - bullish market
            r'⬆️': ' up increase rising growing positive bitcoin ',  # Up arrow - increasing
            
            # Negative sentiment emojis
            r'😢|😭|😞|☹': ' sad down negative bearish ',  # Sad faces
            r'😠|😡|🤬': ' angry upset negative bearish ',  # Angry faces
            r'📉': ' down loss decrease bearish negative bitcoin ',  # Chart down
            r'🐻': ' bear bearish market downtrend negative bitcoin ',  # Bear - bearish market
            r'👎': ' bad negative bearish disapprove bitcoin ',  # Thumbs down
            r'💩': ' bad terrible poor negative bearish bitcoin ',  # Pile of poo - very negative
            r'⬇️': ' down decrease falling dropping negative bitcoin ',  # Down arrow - decreasing
            r'❌': ' error wrong mistake rejected negative bitcoin ',  # Cross mark - error/rejection
            r'⚠️': ' warning caution risk alert negative bitcoin ',  # Warning sign
            r'🔴': ' red loss negative bearish stop bitcoin ',  # Red circle - loss/negative
            r'🆘': ' help emergency distress negative bitcoin ',  # SOS - emergency
            r'🧸': ' bear toy bearish market negative bitcoin ',  # Teddy bear - bearish reference
            r'🗡️': ' knife cut hurt pain negative bitcoin ',  # Dagger - pain/hurt
            r'💣': ' bomb explosion disaster negative bitcoin ',  # Bomb - disaster
            r'🚨': ' alert warning emergency negative bitcoin ',  # Emergency light - warning
            
            # Neutral or context-dependent emojis
            r'❤|♥': ' love strong feeling positive bitcoin ',  # Heart - love (generally positive)
            r'🤔': ' thinking contemplating considering neutral bitcoin ',  # Thinking face - contemplation
            r'🧮': ' calculator math computation analysis bitcoin ',  # Abacus - calculation/analysis
            r'🔍': ' search investigate research explore bitcoin ',  # Magnifying glass - research
            r'⚖️': ' balance scale justice regulation bitcoin ',  # Balance scale - regulation/balance
            r'⏳': ' time waiting patience hourglass bitcoin ',  # Hourglass - waiting/time
            r'🔄': ' refresh reload update cycle bitcoin ',  # Arrows in circle - cycle/repeat
            r'🌐': ' global worldwide international network bitcoin ',  # Globe - global
            r'💱': ' exchange trading swap currency bitcoin ',  # Currency exchange
            r'💼': ' business work professional investment bitcoin ',  # Briefcase - business/work
            r'📱': ' mobile phone technology app bitcoin ',  # Mobile phone - technology
            r'💻': ' computer technology digital bitcoin ',  # Laptop - technology
            r'🖨️': ' printer money printing inflation bitcoin ',  # Printer - money printing
            r'📰': ' news article information bitcoin ',  # Newspaper - news
            r'📊': ' chart graph data analysis bitcoin ',  # Chart - analysis
            r'🧠': ' brain thinking smart intelligent bitcoin ',  # Brain - intelligence/thinking
            r'🌍': ' world global international bitcoin ',  # Globe - global market
            r'🏛️': ' bank government institution regulation bitcoin ',  # Classical building - institution/government
            r'👨‍💼': ' businessman investor trader professional bitcoin ',  # Man office worker
            r'📈📉': ' volatility market swings trading bitcoin ',  # Charts up and down - volatility
            r'🔑': ' key access solution important bitcoin ',  # Key - access/solution
        }
        
        # Enhanced crypto slang dictionary with additional Bitcoin slang terms
        self.slang_map = {
            # Common crypto slang
            r'\bhodl\b': 'hold long-term investment bullish',
            r'\bhodler\b': 'long-term investor bullish',
            r'\bhodling\b': 'holding long-term bullish',
            r'\bhv\b': 'halving bitcoin event bullish',
            r'\bwagmi\b': 'we are going to make it optimistic bullish',
            r'\bngmi\b': 'not going to make it pessimistic bearish',
            r'\bdyor\b': 'do your own research',
            r'\bfomo\b': 'fear of missing out buying high market top',
            r'\bbtfd\b': 'buy the dip opportunity bullish',
            r'\bbtd\b': 'buy the dip opportunity bullish',
            r'\bgm\b': 'good morning',
            r'\bgs\b': 'good ser',
            r'\botc\b': 'over the counter trading',
            r'\bliqd\b': 'liquidated margin call loss',
            r'\brekt\b': 'wrecked loss liquidation bearish',
            
            # Bitcoin-specific slang
            r'\bbtc\b': 'bitcoin',
            r'\bsats\b': 'satoshis bitcoin smallest unit',
            r'\bsatoshi\b': 'smallest bitcoin unit',
            r'\bsatoshis\b': 'smallest bitcoin units',
            r'\bln\b': 'lightning network bitcoin scaling',
            r'\bstack\b': 'accumulate bitcoin bullish',
            r'\bstacking\b': 'accumulating bitcoin regularly bullish',
            r'\btaproot\b': 'bitcoin upgrade improvement',
            r'\bblocksize\b': 'bitcoin block size parameter',
            r'\bcitadel\b': 'bitcoin haven future bullish',
            r'\borange pill\b': 'bitcoin convert bullish',
            r'\bnocoiners\b': 'people without bitcoin',
            r'\bs2f\b': 'stock to flow bitcoin model bullish',
            r'\besg\b': 'environmental social governance bitcoin criticism',
            r'\bfiat\b': 'government currency inflation',
            r'\bwhale\b': 'large bitcoin holder',
            r'\bbearwhale\b': 'large bitcoin seller bearish',
            r'\bminer\b': 'bitcoin network validator',
            r'\bmining\b': 'bitcoin network validation',
            r'\bdifficulty\b': 'bitcoin mining parameter',
            r'\bhashrate\b': 'bitcoin network security measure',
            r'\bhalving\b': 'bitcoin supply reduction event bullish',
            r'\bcold storage\b': 'offline bitcoin wallet secure',
            r'\bhot wallet\b': 'online bitcoin wallet',
            r'\bpaper hands\b': 'weak holder selling bearish',
            r'\bdiamond hands\b': 'strong holder not selling bullish',
            r'\bflippening\b': 'ethereum overtaking bitcoin market cap',
            r'\bmaximalist\b': 'bitcoin only advocate',
            r'\bsuply shock\b': 'reduced bitcoin availability bullish',
            r'\brainbow chart\b': 'bitcoin price logarithmic chart',
            r'\bcoiners\b': 'bitcoin holders',
            r'\bshitcoin\b': 'low value altcoin',
            
            # Other crypto names
            r'\bltc\b': 'litecoin',
            r'\beth\b': 'ethereum',
            r'\bxrp\b': 'ripple',
            r'\bada\b': 'cardano',
            r'\bsol\b': 'solana',
            r'\bbnb\b': 'binance',
            r'\bdot\b': 'polkadot',
            r'\bmatic\b': 'polygon',
            r'\bdoge\b': 'dogecoin',
            r'\bavax\b': 'avalanche',
            r'\blink\b': 'chainlink',
            r'\buni\b': 'uniswap',
            r'\bshib\b': 'shiba inu',
            
            # Trading terms
            r'\bpa\b': 'price action market movement',
            r'\bbgd\b': 'big green dildo price spike bullish',
            r'\bred dildo\b': 'price drop bearish',
            r'\bliq cascade\b': 'liquidation cascade market crash',
            r'\botw\b': 'on the way to moon',
            r'\bretrace\b': 'price retracement pullback',
            r'\bdca\b': 'dollar cost average investment strategy',
            r'\bbotw\b': 'bottom of the week price floor',
            r'\brotw\b': 'rest of the week',
            r'\bcmc\b': 'crypto market cap total value',
            r'\bmc\b': 'market cap total value',
            r'\bdd\b': 'due diligence research',
            r'\bbearish\b': 'expecting price decrease negative',
            r'\bbullish\b': 'expecting price increase positive',
            
            # Market sentiment
            r'\bfud\b': 'fear uncertainty doubt negative',
            r'\bthotw\b': 'the hope of the week',
            r'\bfng\b': 'fear and greed index sentiment',
            r'\bwtf\b': 'what the fork confusion',
            r'\btotm\b': 'top of the market sell signal',
            r'\bbotm\b': 'bottom of the market buy signal',
        }
        
    def _update_vader_lexicon(self):
        """Update VADER lexicon with cryptocurrency-specific terms"""
        # Only perform this if NLTK is available and sia is initialized
        if not NLTK_AVAILABLE or self.sia is None:
            logger.warning("Cannot update VADER lexicon - NLTK not available")
            return
            
        crypto_lexicon = {
            # Positive terms - Enhanced weights and Bitcoin-specific terms
            'hodl': 3.5,  # Increased weight
            'mooning': 4.0,  # Increased weight
            'moon': 3.5,  # Increased weight
            'bullish': 4.0,  # Increased weight
            'bull': 3.5,  # Increased weight
            'breakout': 3.5,  # Increased weight
            'alt season': 2.0,
            'adoption': 3.5,  # Increased weight
            'halving': 4.0,  # Increased for Bitcoin specifically
            'institutional': 3.5,  # Increased weight
            'institutional adoption': 4.5,  # Very strong for Bitcoin
            'institutional investment': 4.5,  # Very strong for Bitcoin
            'partnership': 2.5,
            'rally': 3.0,  # Increased weight
            'support': 2.0,  # Increased weight
            'accumulate': 3.0,  # Increased weight
            'long': 2.0,  # Increased weight
            'undervalued': 3.0,  # Increased weight
            'btc dominance': 3.0,  # Bitcoin-specific
            'digital gold': 4.0,  # Bitcoin-specific
            'store of value': 3.5,  # Bitcoin-specific
            'inflation hedge': 3.5,  # Bitcoin-specific
            'satoshi': 2.0,  # Bitcoin-specific
            'lightning network': 3.5,  # Bitcoin-specific
            'taproot': 3.0,  # Bitcoin-specific
            'segwit': 2.0,  # Bitcoin-specific
            'bitcoin etf': 4.5,  # Bitcoin-specific
            'spot etf': 4.5,  # Bitcoin-specific
            'bitcoin spot etf': 5.0,  # Bitcoin-specific, extremely positive
            'bitcoin futures': 2.5,  # Bitcoin-specific
            'layer 2': 3.0,  # Bitcoin-specific
            'microstrategy': 3.0,  # Bitcoin-specific
            'saylor': 3.0,  # Bitcoin-specific (Michael Saylor)
            'el salvador': 3.0,  # Bitcoin-specific
            'legal tender': 4.0,  # Bitcoin-specific
            'whale accumulation': 3.5,  # Bitcoin-specific
            'record high': 4.0,  # Very positive
            'all time high': 4.0,  # Very positive
            'ath': 4.0,  # Very positive
            'buy the dip': 3.0,  # Strategy
            'btfd': 3.0,  # Buy the F* dip
            'stacking sats': 3.5,  # Bitcoin-specific
            'supply shock': 3.5,  # Bitcoin-specific
            'limited supply': 3.5,  # Bitcoin-specific
            'scarcity': 3.0,  # Bitcoin-specific
            
            # Negative terms - Enhanced weights and Bitcoin-specific terms
            'bearish': -4.0,  # Increased weight
            'bear': -3.0,  # Increased weight
            'dump': -3.5,  # Increased weight
            'dumping': -4.0,  # Increased weight
            'correction': -2.0,  # Increased weight
            'crash': -4.5,  # Increased weight
            'ban': -4.0,  # Increased weight
            'banning': -4.0,  # Increased weight
            'banned': -4.0,  # Increased weight
            'hack': -4.5,  # Increased weight
            'hacked': -4.5,  # Increased weight
            'ponzi': -4.5,  # Increased weight
            'scam': -4.5,  # Increased weight
            'rugpull': -5.0,  # Increased weight
            'rug pull': -5.0,  # Increased weight
            'short': -2.5,  # Increased weight
            'resistance': -1.5,  # Increased weight
            'overvalued': -3.0,  # Increased weight
            'fud': -3.0,  # Increased weight
            'fear': -2.5,  # Increased weight
            'uncertainty': -2.0,  # Increased weight
            'doubt': -2.0,  # Increased weight
            'regulation': -2.0,  # Increased negative for Bitcoin
            'regulatory crackdown': -4.5,  # Bitcoin-specific
            'sec': -2.5,  # Securities and Exchange Commission (increased negative)
            'delisting': -4.0,  # Increased weight
            'manipulation': -3.5,  # Increased weight
            'bubble': -3.5,  # Increased weight
            'sell-off': -3.5,  # Increased weight
            'selloff': -3.5,  # Increased weight
            'whale': -1.5,  # Increased weight
            'whale selling': -3.5,  # Bitcoin-specific
            'double spend': -5.0,  # Bitcoin-specific
            '51% attack': -5.0,  # Bitcoin-specific
            'china ban': -4.0,  # Bitcoin-specific
            'mining ban': -4.0,  # Bitcoin-specific
            'tether risk': -3.5,  # Bitcoin-related
            'leverage liquidation': -4.0,  # Bitcoin-related
            'exchange hack': -5.0,  # Bitcoin-related
            'mt gox': -4.0,  # Bitcoin-specific 
            'energy consumption': -2.5,  # Bitcoin-specific
            'environmental concerns': -3.0,  # Bitcoin-specific
            'carbon footprint': -3.0,  # Bitcoin-specific
            'tax crackdown': -3.5,  # Bitcoin-specific
            'money laundering': -4.0,  # Bitcoin-specific
            'illicit use': -4.0,  # Bitcoin-specific
            'terrorist financing': -5.0,  # Bitcoin-specific
            'stolen bitcoin': -4.5,  # Bitcoin-specific
        }
        
        try:
            # Update the lexicon
            for term, score in crypto_lexicon.items():
                self.sia.lexicon[term] = score
                
            # Add compound terms that require special handling
            # These are checked separately in the analysis logic
            self.bitcoin_specific_terms = {
                'bitcoin halving': 4.5,
                'btc halving': 4.5,
                'bitcoin mining': 1.5,
                'bitcoin mining profitability': 3.0,
                'bitcoin hash rate': 2.0,
                'bitcoin difficulty': 0.0,  # Neutral by default but context matters
                'bitcoin transaction fees': -1.0,  # Slightly negative but depends on context
                'bitcoin mempool': 0.0,  # Neutral but context matters
                'bitcoin network congestion': -2.5,
                'bitcoin etf approval': 5.0,  # Extremely positive
                'bitcoin etf rejection': -4.5,  # Extremely negative
                'bitcoin mass adoption': 4.5,
                'bitcoin institutional buying': 4.5,
                'bitcoin institutional selling': -4.5,
                'bitcoin whale movement': 0.0,  # Neutral but will be analyzed contextually
            }
        except Exception as e:
            logger.error(f"Error updating VADER lexicon: {e}")
            # If we can't update the lexicon, we'll just use the default
    
    def analyze(self, text, context=None):
        """
        Analyze text for sentiment with enhanced cryptocurrency understanding
        
        Args:
            text (str): The text to analyze
            context (dict, optional): Additional context like coin name, source
            
        Returns:
            dict: Sentiment scores and analysis
        """
        if not text or len(text.strip()) < 5:
            return {
                "score": 0,
                "magnitude": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 1,
                "source": "none"
            }
        
        # Preprocess text for better analysis
        cleaned_text = self._preprocess_text(text)
        
        # Check for critical negative terms first - gives higher weight to strongly negative events
        critical_terms = {
            'crash': -0.7,
            'crashed': -0.7,
            'collapse': -0.8, 
            'collapses': -0.8,
            'ban': -0.6, 
            'banned': -0.6,
            'hack': -0.9,
            'hacked': -0.9,
            'scam': -0.8,
            'ponzi': -0.8,
            'rugpull': -0.9,
            'rug pull': -0.9,
            '51% attack': -0.9,
            'double spend': -0.9,
            'stolen': -0.7,
            'theft': -0.7
        }
        
        # Pre-scan for critical negative terms
        critical_neg_score = 0
        critical_terms_found = False
        
        for term, score in critical_terms.items():
            if term in cleaned_text.lower():
                critical_neg_score += score
                critical_terms_found = True
        
        # Check if NLTK is available
        if not NLTK_AVAILABLE or self.sia is None:
            # Fallback to basic keyword analysis with critical term boost
            basic_result = self._basic_keyword_analysis(cleaned_text, context)
            if critical_terms_found:
                # Adjust the score for critical negative terms
                basic_result["score"] = max(-1.0, basic_result["score"] + critical_neg_score)
                basic_result["negative"] = min(1.0, basic_result["negative"] + abs(critical_neg_score)/2)
                basic_result["neutral"] = max(0.0, 1.0 - (basic_result["positive"] + basic_result["negative"]))
                # Recalculate category
                if basic_result["score"] < -0.2:
                    basic_result["category"] = "very negative"
                elif basic_result["score"] < -0.05:
                    basic_result["category"] = "negative"
            return basic_result
        
        try:
            # Get VADER sentiment scores
            vader_scores = self.sia.polarity_scores(cleaned_text)
            compound_score = vader_scores['compound']  # -1 to 1 scale
            
            # Apply coin-specific adjustments if context provided
            if context and 'coin' in context:
                coin_name = context['coin'].lower()
                compound_score = self._apply_coin_specific_adjustments(vader_scores, cleaned_text, coin_name)
            
            # Apply critical negative term adjustments - ensure negative events are weighted properly
            if critical_terms_found:
                # This ensures critical negative news has significant impact
                compound_score = max(-1.0, compound_score + critical_neg_score)
                vader_scores['neg'] = min(1.0, vader_scores['neg'] + abs(critical_neg_score)/2)
                vader_scores['neu'] = max(0.0, 1.0 - (vader_scores['pos'] + vader_scores['neg']))
            
            # Check for specific negative context patterns
            negative_patterns = [
                (r'(crash|collapse|plunge|drop|fall).{0,20}?[0-9]+%', -0.5),  # Price drop percentages
                (r'(ban|bans|banned|banning).{0,30}?(bitcoin|crypto|btc)', -0.6),  # Ban mentions
                (r'(regulation|regulatory).{0,30}?(concern|issue|problem|crackdown)', -0.45),  # Regulatory concerns
                (r'(sec|cftc|regulatory).{0,30}?(reject|denial|lawsuit|sue|investigation)', -0.55),  # Regulatory actions
                (r'(whale|whales).{0,20}?(dump|sell|selling|sold)', -0.4),  # Whale selling
                (r'(exchange|wallet).{0,20}?(hack|breach|compromise|stolen)', -0.8),  # Security issues
                (r'environmental.{0,20}?(concern|issue|problem|impact)', -0.35),  # Environmental concerns
                (r'energy.{0,20}?(usage|consumption|intensive|problem)', -0.3)  # Energy concerns
            ]
            
            # Apply negative pattern matching
            for pattern, adj in negative_patterns:
                if re.search(pattern, cleaned_text.lower()):
                    compound_score = max(-1.0, compound_score + adj)
                    # Also increase the negative component
                    vader_scores['neg'] = min(1.0, vader_scores['neg'] + abs(adj)/2)
                    vader_scores['neu'] = max(0.0, 1.0 - (vader_scores['pos'] + vader_scores['neg']))
            
            # Create magnitude (intensity) based on polarity extremes and text length
            word_count = len(cleaned_text.split())
            magnitude = min(30, max(1.0, abs(compound_score) * word_count * 0.15))
            
            # Determine sentiment category
            if compound_score > 0.2:
                category = "very positive"
            elif compound_score > 0.05:
                category = "positive"
            elif compound_score < -0.2:
                category = "very negative"
            elif compound_score < -0.05:
                category = "negative"
            else:
                category = "neutral"
            
            return {
                "score": compound_score,
                "magnitude": magnitude,
                "positive": vader_scores['pos'],
                "negative": vader_scores['neg'],
                "neutral": vader_scores['neu'],
                "category": category,
                "source": "enhanced_vader"
            }
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            # Fallback to basic keyword analysis
            return self._basic_keyword_analysis(cleaned_text, context)
    
    def _basic_keyword_analysis(self, text, context=None):
        """Basic sentiment analysis using keyword matching when NLTK is unavailable"""
        # Count positive and negative terms
        positive_count = 0
        negative_count = 0
        
        # Check positive terms
        for term, score in self.crypto_pos_terms.items():
            if term in text:
                positive_count += score
        
        # Check negative terms
        for term, score in self.crypto_neg_terms.items():
            if term in text:
                negative_count += abs(score)
        
        # Calculate simple score (-1 to 1)
        total = positive_count + negative_count
        if total == 0:
            score = 0
            positive = 0
            negative = 0
            neutral = 1
        else:
            if positive_count > negative_count:
                score = positive_count / (total * 2)
            elif negative_count > positive_count:
                score = -negative_count / (total * 2)
            else:
                score = 0
            
            # Normalize to match VADER format
            if total > 0:
                positive = positive_count / total
                negative = negative_count / total
            else:
                positive = 0
                negative = 0
            neutral = 1.0 - (positive + negative)
        
        # Simple magnitude calculation based on text length
        word_count = len(text.split())
        magnitude = min(5, max(1.0, abs(score) * word_count * 0.1))
        
        # Determine category
        if score > 0.2:
            category = "very positive"
        elif score > 0.05:
            category = "positive"
        elif score < -0.2:
            category = "very negative"
        elif score < -0.05:
            category = "negative"
        else:
            category = "neutral"
        
        return {
            "score": score,
            "magnitude": magnitude,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "category": category,
            "source": "basic_keyword"
        }
    
    def _preprocess_text(self, text):
        """Clean and prepare text for sentiment analysis"""
        # Convert to lowercase and remove extra whitespace
        text = ' '.join(text.lower().split())
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S*@\S*\s?', '', text)
        
        # Replace emoji patterns with words
        for pattern, replacement in self.emoji_map.items():
            text = re.sub(pattern, replacement, text)
        
        # Replace crypto slang with standard terms
        for pattern, replacement in self.slang_map.items():
            text = re.sub(pattern, replacement, text)
            
        return text
    
    def _replace_emojis(self, text):
        """Replace common emojis with their sentiment meaning"""
        for pattern, replacement in self.emoji_map.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _replace_crypto_slang(self, text):
        """Replace crypto slang with standard terms for better sentiment detection"""
        for pattern, replacement in self.slang_map.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def _apply_coin_specific_adjustments(self, scores, text, coin):
        """Apply coin-specific sentiment adjustments"""
        try:
            compound_score = scores['compound']
            
            # Amplify buy/sell patterns with larger context window
            buy_pattern = re.search(f"(buy|buying|bought|accumulate|long|holding|stack|stacking) .{{0,30}}?{coin}", text)
            sell_pattern = re.search(f"(sell|selling|sold|dump|dumping|short|shorting|liquidate) .{{0,30}}?{coin}", text)
            
            # Apply stronger adjustments
            if buy_pattern:
                if coin == 'bitcoin':
                    compound_score = min(1.0, compound_score + 0.35)  # Stronger boost for Bitcoin
                else:
                    compound_score = min(1.0, compound_score + 0.25)  # Standard boost
            elif sell_pattern:
                if coin == 'bitcoin':
                    compound_score = max(-1.0, compound_score - 0.35)  # Stronger negative for Bitcoin
                else:
                    compound_score = max(-1.0, compound_score - 0.25)  # Standard negative
            
            # Enhanced Bitcoin-specific analysis
            if coin == 'bitcoin' or coin == 'btc':
                # Check for compound Bitcoin terms
                for term, score in self.bitcoin_specific_terms.items():
                    if term in text:
                        # Apply score as a weighted adjustment
                        adjustment = score / 5.0  # Scale to a max of 1.0
                        if score > 0:
                            compound_score = min(1.0, compound_score + adjustment)
                        elif score < 0:
                            compound_score = max(-1.0, compound_score + adjustment)
                
                # Check for Bitcoin price movements with numeric pattern matching
                price_increase_pattern = re.search(r'bitcoin.{0,30}?(up|rise|rose|increase|increased|gained|jumped|surged|rallied).{0,20}?([0-9]+)(%|percent)', text)
                price_decrease_pattern = re.search(r'bitcoin.{0,30}?(down|fall|fell|decrease|decreased|lost|dropped|plunged|declined).{0,20}?([0-9]+)(%|percent)', text)
                
                if price_increase_pattern:
                    # Extract the percentage to apply proportional sentiment
                    try:
                        percentage = float(price_increase_pattern.group(2))
                        # Scale the adjustment: 1% → small, 5% → medium, 10%+ → large
                        adjustment = min(0.4, percentage / 25.0)
                        compound_score = min(1.0, compound_score + adjustment)
                    except:
                        # If parsing fails, apply standard adjustment
                        compound_score = min(1.0, compound_score + 0.15)
                        
                elif price_decrease_pattern:
                    # Extract the percentage to apply proportional sentiment
                    try:
                        percentage = float(price_decrease_pattern.group(2))
                        # Scale the adjustment: 1% → small, 5% → medium, 10%+ → large
                        adjustment = min(0.4, percentage / 25.0)
                        compound_score = max(-1.0, compound_score - adjustment)
                    except:
                        # If parsing fails, apply standard adjustment
                        compound_score = max(-1.0, compound_score - 0.15)
                
                # Advanced context analysis for Bitcoin
                # Check for regulatory context
                if 'regulation' in text or 'regulatory' in text:
                    if 'clarity' in text or 'framework' in text or 'positive' in text:
                        compound_score = min(1.0, compound_score + 0.2)  # Positive regulatory news
                    elif 'crackdown' in text or 'ban' in text or 'restrict' in text:
                        compound_score = max(-1.0, compound_score - 0.3)  # Negative regulatory news
                
                # Check for adoption signals
                adoption_terms = ['adoption', 'accept', 'accepting', 'payment', 'institutional']
                if any(term in text for term in adoption_terms):
                    compound_score = min(1.0, compound_score + 0.25)
                
                # Check for technical indicators
                if 'hash rate' in text:
                    if 'increase' in text or 'higher' in text or 'record' in text:
                        compound_score = min(1.0, compound_score + 0.15)
                    elif 'decrease' in text or 'lower' in text or 'drop' in text:
                        compound_score = max(-1.0, compound_score - 0.15)
                
                # Check for market sentiment indicators
                if 'fear and greed' in text:
                    if 'greed' in text or 'extreme greed' in text:
                        compound_score = min(1.0, compound_score + 0.2)
                    elif 'fear' in text or 'extreme fear' in text:
                        compound_score = max(-1.0, compound_score - 0.2)
                
                # Check for halving context - extremely important for Bitcoin
                if 'halving' in text:
                    if 'approaching' in text or 'upcoming' in text or 'next' in text:
                        compound_score = min(1.0, compound_score + 0.3)  # Future halving is positive
                    elif 'completed' in text or 'happened' in text:
                        compound_score = min(1.0, compound_score + 0.2)  # Recent halving is positive
                
                # Check for ETF context
                if 'etf' in text:
                    if 'approval' in text or 'approved' in text or 'launch' in text:
                        compound_score = min(1.0, compound_score + 0.4)  # ETF approval is very positive
                    elif 'rejected' in text or 'denial' in text or 'denied' in text:
                        compound_score = max(-1.0, compound_score - 0.4)  # ETF rejection is very negative
                
                # Check for whale activity
                if 'whale' in text:
                    if 'accumulation' in text or 'buying' in text or 'accumulating' in text:
                        compound_score = min(1.0, compound_score + 0.2)
                    elif 'selling' in text or 'distribution' in text or 'dumping' in text:
                        compound_score = max(-1.0, compound_score - 0.25)
            
            # Ethereum-specific patterns
            elif coin == 'ethereum' or coin == 'eth':
                if 'eth2' in text or 'eth 2.0' in text or 'merge' in text or 'proof of stake' in text:
                    compound_score = min(1.0, compound_score + 0.2)
                elif 'scaling' in text or 'layer 2' in text or 'l2' in text:
                    compound_score = min(1.0, compound_score + 0.15)
                elif 'gas fees' in text:
                    if 'high' in text or 'expensive' in text:
                        compound_score = max(-1.0, compound_score - 0.15)
                    elif 'low' in text or 'cheaper' in text:
                        compound_score = min(1.0, compound_score + 0.1)
            
            return compound_score
        except Exception as e:
            logger.error(f"Error in coin-specific adjustments: {e}")
            return scores.get('compound', 0) if isinstance(scores, dict) else 0


# Create singleton instance
_analyzer = EnhancedSentimentAnalyzer()

def analyze_sentiment(text, coin=None, source=None):
    """
    Analyze text sentiment using the enhanced analyzer
    
    Args:
        text (str): Text to analyze
        coin (str, optional): Cryptocurrency being discussed
        source (str, optional): Source of the text (twitter, news, etc.)
        
    Returns:
        dict: Sentiment analysis results
    """
    context = {}
    if coin:
        context['coin'] = coin
    if source:
        context['source'] = source
        
    return _analyzer.analyze(text, context)

def batch_analyze(texts, coin=None, source=None):
    """
    Analyze multiple texts and return aggregate sentiment
    
    Args:
        texts (list): List of text strings to analyze
        coin (str, optional): Cryptocurrency being discussed
        source (str, optional): Source of the texts
        
    Returns:
        dict: Aggregate sentiment analysis results
    """
    if not texts:
        return {
            "score": 0,
            "magnitude": 0,
            "positive": 0,
            "negative": 0,
            "neutral": 1,
            "source": "none"
        }
    
    results = [analyze_sentiment(text, coin, source) for text in texts]
    
    # Calculate average scores
    count = len(results)
    avg_score = sum(r["score"] for r in results) / count
    avg_magnitude = sum(r["magnitude"] for r in results) / count
    avg_positive = sum(r["positive"] for r in results) / count
    avg_negative = sum(r["negative"] for r in results) / count
    avg_neutral = sum(r["neutral"] for r in results) / count
    
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
    
    return {
        "score": avg_score,
        "magnitude": avg_magnitude,
        "positive": avg_positive,
        "negative": avg_negative,
        "neutral": avg_neutral,
        "category": category,
        "source": "enhanced_vader_batch",
        "count": count
    }

def store_sentiment_data(coin_id, source_type, sentiment_score, magnitude, 
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
            volume=volume
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
                author=author
            )
            db.session.add(mention)
        
        db.session.commit()
        return True
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to store sentiment data: {e}")
        return False