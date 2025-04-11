from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Access some values
alpaca_key = os.getenv("ALPACA_KEY")
alpaca_endpoint = os.getenv("ALPACA_ENDPOINT")
newsapi_key = os.getenv("NEWSAPI_KEY")

# Print to verify
print(f"Alpaca Key: {alpaca_key}")
print(f"Alpaca Endpoint: {alpaca_endpoint}")
print(f"NewsAPI Key: {newsapi_key}")