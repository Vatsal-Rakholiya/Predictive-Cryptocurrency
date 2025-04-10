"""
Price Prediction Model for Visionx Ai Beginners Cryptocurrency Dashboard
Uses machine learning to predict future cryptocurrency prices.
"""
import os
import logging
import pickle
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from app import db, cache
import api_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

class PricePredictionModel:
    """Machine learning model for cryptocurrency price predictions"""
    
    def __init__(self, coin_id='bitcoin', currency='usd'):
        """Initialize the prediction model"""
        self.coin_id = coin_id
        self.currency = currency
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_file = os.path.join(MODEL_DIR, f'{coin_id}_{currency}_model.pkl')
        self.scaler_file = os.path.join(MODEL_DIR, f'{coin_id}_{currency}_scaler.pkl')
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load a saved model and scaler if they exist"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.scaler_file):
                self.model = joblib.load(self.model_file)
                self.scaler = joblib.load(self.scaler_file)
                logger.info(f"Loaded existing model for {self.coin_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _save_model(self):
        """Save the trained model and scaler to disk"""
        try:
            if self.model:
                joblib.dump(self.model, self.model_file)
                joblib.dump(self.scaler, self.scaler_file)
                logger.info(f"Model saved to {self.model_file}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def _prepare_features(self, data):
        """
        Prepare features for prediction from historical price data
        
        Creates features:
        - Price changes over different time periods
        - Moving averages
        - Volatility measures
        - Day of week, hour of day
        """
        try:
            # Ensure data is a DataFrame with timestamps
            if isinstance(data, dict):
                # Convert API response to DataFrame
                prices = data.get('prices', [])
                if not prices or len(prices) < 24:
                    logger.error("Not enough price data points to prepare features")
                    return pd.DataFrame()  # Return empty dataframe instead of raising error
                
                df = pd.DataFrame(prices, columns=['timestamp', 'price'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                logger.error("Invalid data format for feature preparation")
                return pd.DataFrame()  # Return empty dataframe
            
            # Ensure chronological order
            df = df.sort_index()
            
            # Basic features
            df['price_1h_change'] = df['price'].pct_change(periods=1)
            df['price_24h_change'] = df['price'].pct_change(periods=24)
            df['price_7d_change'] = df['price'].pct_change(periods=168)  # 24*7 hours
            
            # Moving averages
            df['ma_6h'] = df['price'].rolling(window=6).mean()
            df['ma_24h'] = df['price'].rolling(window=24).mean()
            df['ma_7d'] = df['price'].rolling(window=168).mean()
            
            # Volatility (standard deviation)
            df['volatility_24h'] = df['price'].rolling(window=24).std()
            
            # Time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            
            # Technical indicators
            # RSI (Relative Strength Index) - simplified
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            loss = loss.replace(0, 0.0001)  # Avoid division by zero
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values (from the rolling calculations)
            df.dropna(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()  # Return empty dataframe
    
    def _create_sequences(self, df, sequence_length=24):
        """
        Create input sequences for prediction
        Each sequence uses 'sequence_length' hours of data to predict the next hour
        """
        features = df.drop(columns=['price']).values
        target = df['price'].values
        
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, days=30):
        """
        Train the prediction model using historical data
        
        Args:
            days (int): Number of days of historical data to use for training
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Use CSV data for Ethereum
            if self.coin_id == 'ethereum':
                return self._train_with_csv()
            else:
                return self._train_with_api(days)
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def _train_with_csv(self):
        """Train the model using CSV data for Ethereum"""
        try:
            logger.info(f"Using CSV data for Ethereum price prediction")
            csv_path = 'attached_assets/Ethereum Historical Data.csv'
            
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return False
                
            # Read the CSV file
            df_raw = pd.read_csv(csv_path)
            
            # Convert date format and create proper timestamp
            df_raw['timestamp'] = pd.to_datetime(df_raw['Date'], format='%m/%d/%Y')
            df_raw = df_raw.rename(columns={'Price': 'price'})
            df_raw = df_raw[['timestamp', 'price']]
            df_raw.set_index('timestamp', inplace=True)
            df_raw = df_raw.sort_index()
            
            # For daily data, create custom features
            df = df_raw.copy()
            
            # Basic features - adjusted for daily data
            df['price_1d_change'] = df['price'].pct_change(periods=1)
            df['price_7d_change'] = df['price'].pct_change(periods=7)
            df['price_30d_change'] = df['price'].pct_change(periods=30)
            
            # Moving averages for daily data
            df['ma_3d'] = df['price'].rolling(window=3).mean()
            df['ma_7d'] = df['price'].rolling(window=7).mean()
            df['ma_30d'] = df['price'].rolling(window=30).mean()
            
            # Volatility features
            df['volatility_7d'] = df['price'].rolling(window=7).std()
            df['volatility_30d'] = df['price'].rolling(window=30).std()
            
            # Time features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            # Technical indicators
            # RSI (Relative Strength Index) - simplified
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            loss = loss.replace(0, 0.0001)  # Avoid division by zero
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values (from the rolling calculations)
            df.dropna(inplace=True)
            
            # Use a smaller sequence length for daily data
            sequence_length = 7  # Use 7 days of data to predict the next day
            
            if len(df) < sequence_length + 1:  # Need enough data points
                logger.error(f"Not enough data points in CSV after processing: {len(df)}")
                return False
            
            # Create sequences with the smaller length
            features = df.drop(columns=['price']).values
            target = df['price'].values
            
            X, y = [], []
            for i in range(len(df) - sequence_length):
                X.append(features[i:i+sequence_length])
                y.append(target[i+sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            if len(X) < 10:  # Need enough sequences
                logger.error(f"Not enough sequences for training: {len(X)}")
                return False
            
            # Scale features
            X_reshaped = X.reshape(X.shape[0], -1)  # Flatten sequences for scaling
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)  # Reshape back to sequences
            
            # Train model
            # Using RandomForestRegressor for robustness
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled.reshape(X_scaled.shape[0], -1), y)
            
            # Save model
            self._save_model()
            
            logger.info(f"Model trained successfully with {len(df)} data points from CSV")
            return True
        
        except Exception as e:
            logger.error(f"Error training model with CSV: {e}")
            return False
    
    def _train_with_api(self, days=30):
        """
        Train the prediction model using API data
        
        Args:
            days (int): Number of days of historical data to use for training
        
        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            # Fetch historical data
            logger.info(f"Fetching {days} days of historical data for {self.coin_id}")
            chart_data = api_service.get_coin_market_chart(self.coin_id, self.currency, str(days))
            
            if not chart_data or 'prices' not in chart_data:
                logger.error("Failed to fetch historical data")
                return False
            
            # Prepare features
            df = self._prepare_features(chart_data)
            if len(df) < 48:  # Need at least 2 days of data
                logger.error(f"Not enough data points: {len(df)}")
                return False
            
            # Create sequences
            X, y = self._create_sequences(df)
            
            # Scale features
            X_reshaped = X.reshape(X.shape[0], -1)  # Flatten sequences for scaling
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X_scaled = X_scaled.reshape(X.shape)  # Reshape back to sequences
            
            # Train model
            # Using RandomForestRegressor for robustness
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled.reshape(X_scaled.shape[0], -1), y)
            
            # Save model
            self._save_model()
            
            logger.info(f"Model trained successfully with {len(df)} data points")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    @cache.memoize(timeout=3600)  # Cache predictions for 1 hour to reduce API calls
    def predict_next_hours(self, hours=24):
        """
        Predict prices for the next X hours
        
        Args:
            hours (int): Number of hours to predict into the future
            
        Returns:
            dict: Predicted prices with timestamps
        """
        try:
            if not self.model:
                # Try to load model from disk first
                if not self._load_model():
                    # If no model on disk, try to train
                    if not self.train():
                        logger.error("No model available and training failed")
                        return None
            
            # For Ethereum, use CSV data
            if self.coin_id == 'ethereum':
                return self._predict_with_csv(hours)
            else:
                return self._predict_with_api(hours)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
            
    def _predict_with_csv(self, hours=24):
        """Make predictions using CSV data for Ethereum"""
        try:
            csv_path = 'attached_assets/Ethereum Historical Data.csv'
            
            if not os.path.exists(csv_path):
                logger.error(f"CSV file not found: {csv_path}")
                return None
                
            # Read the CSV file
            df_raw = pd.read_csv(csv_path)
            
            # Convert date format and create proper timestamp
            df_raw['timestamp'] = pd.to_datetime(df_raw['Date'], format='%m/%d/%Y')
            df_raw = df_raw.rename(columns={'Price': 'price'})
            df_raw = df_raw[['timestamp', 'price']]
            df_raw.set_index('timestamp', inplace=True)
            df_raw = df_raw.sort_index()
            
            # Get the most recent data
            df_raw = df_raw.tail(100)  # Use last 100 days
            
            # For daily data, create custom features - same as in _train_with_csv
            df = df_raw.copy()
            
            # Basic features - adjusted for daily data
            df['price_1d_change'] = df['price'].pct_change(periods=1)
            df['price_7d_change'] = df['price'].pct_change(periods=7)
            df['price_30d_change'] = df['price'].pct_change(periods=30)
            
            # Moving averages for daily data
            df['ma_3d'] = df['price'].rolling(window=3).mean()
            df['ma_7d'] = df['price'].rolling(window=7).mean()
            df['ma_30d'] = df['price'].rolling(window=30).mean()
            
            # Volatility features
            df['volatility_7d'] = df['price'].rolling(window=7).std()
            df['volatility_30d'] = df['price'].rolling(window=30).std()
            
            # Time features
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            
            # Technical indicators
            # RSI (Relative Strength Index) - simplified
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            loss = loss.replace(0, 0.0001)  # Avoid division by zero
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Drop NaN values (from the rolling calculations)
            df.dropna(inplace=True)
            
            # Use a smaller sequence length for daily data
            sequence_length = 7  # Use 7 days to predict the next day
            
            if len(df) < sequence_length:
                logger.error("Not enough data points in CSV for prediction")
                return None
            
            # Get the most recent sequence
            latest_sequence = df.drop(columns=['price']).values[-sequence_length:]
            
            # Make predictions iteratively for each future day (converting hours to days)
            predictions = []
            current_time = df.index[-1]
            latest_price = df['price'].iloc[-1]
            sequence = latest_sequence.copy()
            
            for i in range(hours):
                # Scale the sequence
                sequence_flat = sequence.reshape(1, -1)
                sequence_scaled = self.scaler.transform(sequence_flat)
                
                # Predict next price
                next_price = self.model.predict(sequence_scaled)[0]
                
                # For daily data, increment by day instead of hour
                current_time = current_time + timedelta(days=1)
                
                predictions.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'price': float(next_price),
                    'is_prediction': True
                })
                
                # Update sequence for next prediction (rolling window)
                # Remove oldest row's data and add new prediction
                new_row = sequence[-1].copy()  # Copy the last row features
                # For simplicity, we'll just shift all the data
                sequence = np.vstack([sequence[1:], new_row])
            
            # Return historical and predicted prices
            historical = [{
                'timestamp': str(idx),
                'price': float(price),
                'is_prediction': False
            } for idx, price in zip(df.index[-7:], df['price'].values[-7:])]
            
            return {
                'historical': historical,
                'predictions': predictions,
                'current_price': float(latest_price)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with CSV: {e}")
            return None
            
    def _predict_with_api(self, hours=24):
        """Make predictions using API data"""
        try:
            # Fetch recent data for prediction with longer cache
            chart_data = api_service.get_coin_market_chart(self.coin_id, self.currency, '7')
            if not chart_data:
                logger.error("Failed to fetch recent data for prediction")
                return None
            
            # Prepare features
            df = self._prepare_features(chart_data)
            if len(df) < 24:
                logger.error("Not enough recent data for prediction")
                return None
            
            # Get the most recent sequence
            latest_sequence = df.drop(columns=['price']).values[-24:]
            
            # Make predictions iteratively for each future hour
            predictions = []
            current_time = df.index[-1]
            latest_price = df['price'].iloc[-1]
            sequence = latest_sequence.copy()
            
            for i in range(hours):
                # Scale the sequence
                sequence_flat = sequence.reshape(1, -1)
                sequence_scaled = self.scaler.transform(sequence_flat)
                
                # Predict next price
                next_price = self.model.predict(sequence_scaled)[0]
                
                # Add prediction to results
                current_time = current_time + timedelta(hours=1)
                predictions.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'price': float(next_price),
                    'is_prediction': True
                })
                
                # Update sequence for next prediction (rolling window)
                # Remove oldest hour's data and add new prediction
                new_row = sequence[-1].copy()  # Copy the last row features
                # Update any price-dependent features in new_row based on prediction
                # For simplicity, we'll just shift all the data
                sequence = np.vstack([sequence[1:], new_row])
            
            # Return historical and predicted prices
            historical = [{
                'timestamp': str(idx),
                'price': float(price),
                'is_prediction': False
            } for idx, price in zip(df.index[-24:], df['price'].values[-24:])]
            
            return {
                'historical': historical,
                'predictions': predictions,
                'current_price': float(latest_price)
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with API: {e}")
            return None

# Initialize with a singleton instance
prediction_service = PricePredictionModel()

# Function to get or create a model for a specific coin
def get_prediction_model(coin_id='bitcoin', currency='usd'):
    """
    Get or create a prediction model for a specific coin
    
    Args:
        coin_id (str): Cryptocurrency ID (e.g. 'bitcoin')
        currency (str): Base currency for price (e.g. 'usd')
    
    Returns:
        PricePredictionModel: Model instance for the specified coin
    """
    if coin_id == 'bitcoin' and currency == 'usd':
        return prediction_service
    
    return PricePredictionModel(coin_id=coin_id, currency=currency)