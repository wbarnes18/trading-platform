import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# List of stocks
stocks = ["SPY", "TSLA", "NVDA"]

for stock in stocks:
    print(f"Training model for {stock}...")
    
    # Step 1: Load the data
    df = pd.read_csv(f"data/{stock.lower()}_data_with_indicators.csv", header=[0, 1, 2])
    # Flatten the multi-level header
    new_columns = []
    for col in df.columns:
        if col[2] == 'Date':
            new_columns.append('Date')
        elif col[0] == 'Price':
            new_columns.append(col[1])
        else:
            new_columns.append(col[0])
    df.columns = new_columns
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Step 2: Create the target variable (price direction)
    # 1 if the next day's close is higher, 0 if lower
    df['Price_Direction'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    # Drop the last row since it won't have a target
    df = df[:-1]

    # Step 3: Select features and target
    features = ['Close', 'SMA_20', 'RSI', 'MACD', 'MACD_Signal']
    X = df[features].fillna(0)  # Fill NaN values with 0 (e.g., early rows where indicators aren't calculated yet)
    y = df['Price_Direction']

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Step 5: Train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Evaluate the model
    y_pred = model.predict(X_test)
    print(f"Results for {stock}:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

    # Step 7: Save the model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{stock.lower()}_rf_model.pkl")
    print(f"Model saved to models/{stock.lower()}_rf_model.pkl\n")

print("Training complete for all stocks.")