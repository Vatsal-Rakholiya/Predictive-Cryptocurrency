from app import db
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import enum

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    preferred_currency = db.Column(db.String(10), default="USD")
    dark_mode = db.Column(db.Boolean, default=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    watchlist_items = db.relationship('WatchlistItem', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    featured_coins = db.relationship('FeaturedCoin', backref='admin', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class WatchlistItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    coin_id = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<WatchlistItem {self.coin_id}>'
    
    # Create a unique constraint to prevent duplicate watchlist entries
    __table_args__ = (db.UniqueConstraint('coin_id', 'user_id', name='unique_coin_user'),)

# New models for sentiment analysis

class DataSourceType(enum.Enum):
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    GENERAL = "general"

class SentimentRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    coin_id = db.Column(db.String(50), nullable=False, index=True)
    source = db.Column(db.Enum(DataSourceType), nullable=False)
    sentiment_score = db.Column(db.Float, nullable=False)  # -1.0 to 1.0 score
    magnitude = db.Column(db.Float, nullable=True)  # Intensity of sentiment (0.0 to +inf)
    volume = db.Column(db.Integer, default=1)  # Number of posts/mentions analyzed
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def __repr__(self):
        return f'<SentimentRecord {self.coin_id} from {self.source.value}: {self.sentiment_score}>'

class SentimentMention(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    coin_id = db.Column(db.String(50), nullable=False, index=True)
    source = db.Column(db.Enum(DataSourceType), nullable=False)
    content = db.Column(db.Text, nullable=False)  # The mention text (tweet, post, etc)
    sentiment_score = db.Column(db.Float, nullable=False)
    url = db.Column(db.String(512), nullable=True)  # Link to original content if available
    author = db.Column(db.String(128), nullable=True)  # Author if available
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SentimentMention {self.coin_id}: {self.sentiment_score}>'
        
    # Index for faster retrieval of recent mentions
    __table_args__ = (db.Index('idx_mentions_coin_time', 'coin_id', 'created_at'),)

class FeaturedCoin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    coin_id = db.Column(db.String(50), nullable=False, index=True, unique=True)
    display_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    position = db.Column(db.Integer, default=0)  # For controlling the display order
    is_active = db.Column(db.Boolean, default=True)
    added_by = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<FeaturedCoin {self.coin_id}: {self.display_name}>'
