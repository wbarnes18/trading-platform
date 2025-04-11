import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Explicitly specify the path to the .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Load the NewsAPI key
newsapi_key = os.getenv("NEWSAPI_KEY")
if not newsapi_key:
    print("Error: NEWSAPI_KEY not found in .env file.")
    exit(1)
print(f"Loaded NewsAPI key: {newsapi_key}")

# Step 1: Fetch historical data for multiple stocks
stocks = ["SPY", "TSLA", "NVDA"]
start_date = "2015-04-01"
end_date = "2025-04-01"

for stock in stocks:
    print(f"Processing {stock}...")
    data = yf.download(stock, start=start_date, end=end_date, interval="1d")
    if data.empty:
        print(f"Error: No data downloaded for {stock}. Check date range or internet connection.")
        continue
    data.to_csv(f"data/{stock.lower()}_data.csv")

    # Step 2: Compute technical indicators
    close_series = data["Close"][stock] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
    if close_series.isna().all():
        print(f"Error: Close column for {stock} contains all NaN values. Cannot compute indicators.")
        continue
    close_series = close_series.ffill()
    data["SMA_20"] = ta.sma(close_series, length=20)
    data["RSI"] = ta.rsi(close_series, length=14)
    macd = ta.macd(close_series, fast=12, slow=26, signal=9)
    if macd is not None:
        print(f"MACD columns for {stock}:", macd.columns)
        data["MACD"] = macd.get("MACD_12_26_9", 0)
        data["MACD_Signal"] = macd.get("MACD_signal_12_26_9", macd.get("MACDs_12_26_9", 0))
    else:
        print(f"MACD calculation failed for {stock}, setting to 0")
        data["MACD"] = 0
        data["MACD_Signal"] = 0
    data.to_csv(f"data/{stock.lower()}_data_with_indicators.csv")

    # Step 3: Fetch news from NewsAPI
    try:
        # Set date range for the last 30 days (free tier limitation)
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={newsapi_key}&language=en&sortBy=publishedAt&from={from_date}&pageSize=10"
        print(f"Fetching news for {stock} from URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json().get("articles", [])
        if not news_data:
            print(f"No articles found for {stock} in NewsAPI response.")
    except requests.RequestException as e:
        print(f"Error fetching news for {stock} from NewsAPI: {e}")
        print(f"Response status code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
        news_data = [{"title": f"Placeholder News for {stock}", "publishedAt": "2025-04-01T00:00:00Z"}]

    analyzer = SentimentIntensityAnalyzer()
    news_with_sentiment = []
    for article in news_data:
        title = article.get("title", "No Title")
        published_at = article.get("publishedAt", "2025-04-01T00:00:00Z")
        sentiment = analyzer.polarity_scores(title)
        news_with_sentiment.append({
            "title": title,
            "date": published_at,
            "sentiment": sentiment["compound"]
        })
    news_df = pd.DataFrame(news_with_sentiment)
    news_df.to_csv(f"data/{stock.lower()}_news.csv")

    # Step 4: Compute NSMI (News Sentiment Momentum Indicator)
    if not news_df.empty and "date" in news_df.columns and "sentiment" in news_df.columns:
        # Convert date to date-only format (e.g., "2025-04-10")
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.strftime("%Y-%m-%d")
        
        # Group by date and calculate daily average sentiment
        daily_sentiment = news_df.groupby("date")["sentiment"].mean()
        sentiment_df = pd.DataFrame(daily_sentiment).reset_index()
        
        # Calculate sentiment momentum (difference over 5 days)
        sentiment_df["sentiment_momentum"] = sentiment_df["sentiment"].diff(periods=5)
        
        # Compute NSMI
        if sentiment_df["sentiment_momentum"].dropna().empty:
            # Not enough days for momentum, set NSMI to 0
            print(f"Not enough data to compute NSMI for {stock}, setting NSMI to 0")
            sentiment_df["NSMI"] = 0.0
        else:
            # Scale momentum to NSMI (0-100)
            min_momentum = sentiment_df["sentiment_momentum"].min()
            max_momentum = sentiment_df["sentiment_momentum"].max()
            if min_momentum == max_momentum:
                sentiment_df["NSMI"] = 50.0  # Neutral if no variation
            else:
                sentiment_df["NSMI"] = (sentiment_df["sentiment_momentum"] - min_momentum) / (max_momentum - min_momentum) * 100
    else:
        print(f"No valid news data for {stock}, setting NSMI to 0")
        sentiment_df = pd.DataFrame({
            "date": ["2025-04-01"],
            "sentiment": [0.0],
            "sentiment_momentum": [0.0],
            "NSMI": [0.0]
        })
    sentiment_df.to_csv(f"data/{stock.lower()}_nsmi.csv", index=False)

print("Data collection complete. Files saved in data/")