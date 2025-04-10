# Visionx Ai Beginners Cryptocurrency Dashboard

## Overview

Visionx Ai Beginners is a comprehensive cryptocurrency analytics platform built with Python Flask that provides real-time market data, advanced sentiment analysis, and machine learning-based price predictions. The application is designed to help cryptocurrency investors and enthusiasts make more informed decisions by combining multiple data sources with advanced analytics techniques.

![Visionx Ai Beginners Dashboard](https://via.placeholder.com/1200x600?text=Visionx+Ai+Beginners+Cryptocurrency+Dashboard)

## Key Features

### Real-time Market Data
- Top 50 cryptocurrencies by market cap
- Detailed coin information and price history
- Global market statistics and trends
- Trending coins highlighting market momentum
- Multiple currency support for price conversion

### Advanced Sentiment Analysis
- Natural Language Processing with NLTK and TextBlob
- Cryptocurrency-specific lexicon enhancements
- Multi-source sentiment collection (news, social media)
- Emoji and crypto slang recognition
- Interactive sentiment analysis tool

### Price Prediction
- Machine learning-based price forecasting
- Historical data analysis with multiple features
- Sequence-based prediction models
- Specialized Ethereum forecasting
- Sentiment-enhanced direction indicators

### User Features
- Secure user authentication and account management
- Customizable watchlists
- Dark/light mode preference settings
- Currency preferences
- Responsive design for all devices

## Technology Stack

### Backend
- Python Flask web framework
- SQLAlchemy ORM with PostgreSQL database
- NLTK and TextBlob for Natural Language Processing
- Scikit-learn for machine learning models
- CoinGecko API for cryptocurrency data
- NewsAPI for sentiment data sources

### Frontend
- HTML5/CSS3/JavaScript
- Bootstrap 5 for responsive design
- Chart.js for interactive data visualization
- Font Awesome for icons
- Modern responsive UI with dark/light modes

## Installation

### Prerequisites
- Python 3.7+
- PostgreSQL database
- Node.js and npm (for frontend assets)

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/visionx-crypto-dashboard.git
cd visionx-crypto-dashboard
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Required environment variables
export DATABASE_URL="postgresql://username:password@localhost/visionx"
export SECRET_KEY="your-secret-key"
export NEWS_API_KEY="your-newsapi-key"
```

5. Initialize the database:
```bash
flask db upgrade
```

6. Run the application:
```bash
flask run
```

The application will be available at http://localhost:5000

## Usage

### Market Overview
The home page displays the top cryptocurrencies by market cap, global market statistics, and trending coins. Click on any coin to see detailed information, price history, and analytics.

### Sentiment Analysis
Navigate to the Sentiment Analysis section to see market sentiment for various cryptocurrencies. The system analyzes news and social media sentiment to provide insights into market mood. Use the Enhanced Analyzer to test custom text sentiment.

### Price Prediction
The Price Prediction section provides machine learning-based price forecasts for major cryptocurrencies. The Ethereum Forecast page offers specialized predictions for Ethereum with enhanced accuracy.

### User Features
Register for an account to create a custom watchlist, set your preferred currency, and choose between dark and light modes.

## Documentation

For detailed documentation, see [VisionX_Detailed_Documentation.md](VisionX_Detailed_Documentation.md) (to be renamed to Visionx_Ai_Beginners_Documentation.md).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [CoinGecko API](https://www.coingecko.com/en/api) for cryptocurrency market data
- [NewsAPI](https://newsapi.org/) for news sentiment data
- [NLTK](https://www.nltk.org/) for natural language processing
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [Bootstrap](https://getbootstrap.com/) for the frontend framework
- [Chart.js](https://www.chartjs.org/) for interactive charts"# Predictive-Cryptocurrency" 
