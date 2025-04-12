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

# Load .env file if it exists
env_path = os.path.join(os.path.dirname(__file__), '.env')
print(f"Attempting to load .env file from: {env_path}")
if os.path.exists(env_path):
    load_dotenv(env_path)
    print("Loaded .env file successfully")
else:
    print(".env file not found, relying on environment variables")

# Load the NewsAPI key
newsapi_key = os.getenv("NEWSAPI_KEY")
if not newsapi_key:
    print("Error: NEWSAPI_KEY not found in environment variables. Please set it in Render's environment settings.")
    exit(1)
print(f"Loaded NewsAPI key: {newsapi_key}")

# Initialize the Dash app and define server
app = dash.Dash(__name__)
server = app.server  # Explicitly define server for Gunicorn

# List of stocks
stocks = ["SPY", "TSLA", "NVDA"]

# Function to fetch and process data
def fetch_data(stock):
    # Fetch historical data
    data = yf.download(stock, start="2015-04-01", end="2025-04-11", interval="1d")
    if data.empty:
        return None
    # (Rest of your fetch_data function remains unchanged)
    # Compute technical indicators, fetch news, compute NSMI, etc.
    # Return data, news_df, sentiment_df as before

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

# Define layout
app.layout = html.Div(
    # Your layout code here
)

# Callback to update the dashboard
@app.callback(
    Output("dashboard-content", "children"),
    [Input("stock-dropdown", "value"), Input("interval-component", "n_intervals")]
)
def update_dashboard(selected_stock, n_intervals):
    # Your callback code here

    if __name__ == "__main__":
        port = int(os.getenv("PORT", 8050))
        app.run(debug=False, host="0.0.0.0", port=port)