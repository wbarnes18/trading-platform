import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import joblib
import numpy as np
import yfinance as yf
import pandas_ta as ta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

# Load .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Loading .env file from: {env_path}")
load_dotenv(env_path)

# Load the NewsAPI key
newsapi_key = os.getenv("NEWSAPI_KEY")
if not newsapi_key:
    print("Error: NEWSAPI_KEY not found in .env file.")
    exit(1)
print(f"Loaded NewsAPI key: {newsapi_key}")

# Initialize the Dash app
app = dash.Dash(__name__)

# List of stocks
stocks = ["SPY", "TSLA", "NVDA"]

# Function to fetch and process data
def fetch_data(stock):
    # Fetch historical data
    data = yf.download(stock, start="2015-04-01", end="2025-04-11", interval="1d")
    if data.empty:
        return None

    # Compute technical indicators
    close_series = data["Close"][stock] if isinstance(data["Close"], pd.DataFrame) else data["Close"]
    if close_series.isna().all():
        return None
    close_series = close_series.ffill()
    data["SMA_20"] = ta.sma(close_series, length=20)
    data["RSI"] = ta.rsi(close_series, length=14)
    macd = ta.macd(close_series, fast=12, slow=26, signal=9)
    if macd is not None:
        data["MACD"] = macd.get("MACD_12_26_9", 0)
        data["MACD_Signal"] = macd.get("MACD_signal_12_26_9", macd.get("MACDs_12_26_9", 0))
    else:
        data["MACD"] = 0
        data["MACD_Signal"] = 0

    # Fetch news from NewsAPI
    try:
        from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={newsapi_key}&language=en&sortBy=publishedAt&from={from_date}&pageSize=10"
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json().get("articles", [])
        if not news_data:
            print(f"No articles found for {stock} in NewsAPI response.")
    except requests.RequestException as e:
        print(f"Error fetching news for {stock} from NewsAPI: {e}")
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

    # Compute NSMI
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
        sentiment_df = pd.DataFrame({
            "date": ["2025-04-01"],
            "sentiment": [0.0],
            "sentiment_momentum": [0.0],
            "NSMI": [0.0]
        })

    return data, news_df, sentiment_df

# Load initial data and models
data_dict = {}
news_dict = {}
nsmi_dict = {}
model_dict = {}
for stock in stocks:
    result = fetch_data(stock)
    if result is None:
        print(f"Failed to fetch data for {stock}")
        continue
    data_dict[stock], news_dict[stock], nsmi_dict[stock] = result

    # Load the trained model
    try:
        model_dict[stock] = joblib.load(f"models/{stock.lower()}_rf_model.pkl")
    except Exception as e:
        print(f"Error loading model for {stock}: {e}")
        model_dict[stock] = None

# Layout
app.layout = html.Div(
    style={"backgroundColor": "#1a1a1a", "height": "100vh", "padding": "10px"},
    children=[
        html.H1("Trading Dashboard", style={"color": "#ffffff", "fontFamily": "Roboto"}),
        dcc.Dropdown(
            id="stock-dropdown",
            options=[{"label": stock, "value": stock} for stock in stocks],
            value="SPY",
            style={"width": "200px", "color": "#000000"}
        ),
        dcc.Interval(
            id="interval-component",
            interval=5*60*1000,  # Update every 5 minutes (in milliseconds)
            n_intervals=0
        ),
        html.Div(id="dashboard-content")
    ]
)

# Callback to update the dashboard based on the selected stock and interval
@app.callback(
    Output("dashboard-content", "children"),
    [Input("stock-dropdown", "value"), Input("interval-component", "n_intervals")]
)
def update_dashboard(selected_stock, n_intervals):
    # Refresh data for the selected stock
    result = fetch_data(selected_stock)
    if result is None:
        return html.Div("Error fetching data for selected stock.", style={"color": "#ffffff"})
    data_dict[selected_stock], news_dict[selected_stock], nsmi_dict[selected_stock] = result

    data = data_dict[selected_stock]
    news_df = news_dict[selected_stock]
    nsmi_df = nsmi_dict[selected_stock]
    model = model_dict[selected_stock]
    car_count = 150  # Placeholder from satellite data

    # Make a prediction for the next day's price direction
    prediction = "Not Available"
    if model is not None:
        latest_data = data.tail(1)[['Close', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']].fillna(0)
        if not latest_data.empty:
            pred = model.predict(latest_data)[0]
            prediction = "Up" if pred == 1 else "Down"

    # Create subplots: Price, RSI, MACD, NSMI
    fig = sp.make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                           subplot_titles=("Price", "RSI", "MACD", "NSMI"),
                           row_heights=[0.4, 0.2, 0.2, 0.2])

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index, open=data["Open"], high=data["High"],
                                 low=data["Low"], close=data["Close"], name="OHLC"), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["SMA_20"], name="SMA 20", line=dict(color="#00ff00")), row=1, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="#ff9900")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="#00ccff")), row=3, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data["MACD_Signal"], name="Signal", line=dict(color="#ff3399")), row=3, col=1)

    # NSMI
    fig.add_trace(go.Scatter(x=nsmi_df["date"], y=nsmi_df["NSMI"], name="NSMI", line=dict(color="#ff00ff")), row=4, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="green", row=4, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="red", row=4, col=1)

    # Styling
    fig.update_layout(
        title=f"{selected_stock} Trading Dashboard",
        template="plotly_dark",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(family="Roboto", size=12, color="#ffffff"),
        showlegend=True,
        height=1000
    )
    fig.update_xaxes(rangeslider_visible=False)

    return html.Div(
        style={"display": "flex", "flexDirection": "row"},
        children=[
            # Main Chart
            html.Div(
                dcc.Graph(id="main-chart", figure=fig),
                style={"width": "70%"}
            ),
            # Side Panel
            html.Div(
                style={"width": "30%", "padding": "10px", "backgroundColor": "#2a2a2a", "borderRadius": "5px"},
                children=[
                    html.H3("Supply Chain Signals", style={"color": "#ffffff"}),
                    html.P(f"Foxconn Factory Activity (Car Count): {car_count}", style={"color": "#00ff00"}),
                    html.P("Interpretation: High activity suggests strong production.", style={"color": "#ffffff"}),
                    html.Hr(),
                    html.H3("Price Prediction", style={"color": "#ffffff"}),
                    html.P(f"Tomorrow's Direction: {prediction}", style={"color": "#00ff00" if prediction == "Up" else "#ff0000"}),
                    html.Hr(),
                    html.H3("News Feed", style={"color": "#ffffff"}),
                    html.Ul([
                        html.Li(f"{row['title']} (Sentiment: {row['sentiment']:.2f})", style={"color": "#ffffff"})
                        for _, row in news_df.head(5).iterrows()
                    ])
                ]
            )
        ]
    )

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)