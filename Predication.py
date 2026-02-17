import yfinance as yf
import joblib
import pandas as pd
from Data_spliting import test_train_divide
from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing
from Features import Feature_data
import matplotlib.pyplot as plt

df = yf.download("TCS.NS", start="2024-01-01", end="2024-12-23")
model = joblib.load("stock_model_v2.pkl")
model_features = model.feature_names_in_

validate_stock_data(df)
df = proper_preprocessing(df)

dp = df.copy()
df = Feature_data(df)
data = test_train_divide(df)
print(df)
print("Model Features:", model_features)
future_predictions = []
df_copy = df.copy()





def predict_future(df, model, days=4):  
    for i in range(days):   # predict next 4 days
        
        # Take last available row
        last_row = df_copy.iloc[-1:]
        X = last_row[model_features]
        
        prob = model.predict_proba(X)[:, 1][0]
        pred = 1 if prob > 0.45 else 0
        
        future_predictions.append({
            "Day": i+1,
            "Prediction": pred,
            "Probability": prob
        })
        
        # --- Simulate next day close ---
        
        last_close = df_copy["Adj Close"].iloc[-1]
        
        # If prediction = 1 assume small positive move
        # If 0 assume small negative move
        simulated_return = 0.005 if pred == 1 else -0.005
        
        new_close = last_close * (1 + simulated_return)
        
        # Create new row
        new_row = df_copy.iloc[-1:].copy()
        new_row["Adj Close"] = new_close
        
        # Append and recompute features
        df_copy = pd.concat([df_copy, new_row])
        
        # Recalculate indicators
        df_copy["Return"] = df_copy["Adj Close"].pct_change()
        df_copy["MA_5"] = df_copy["Adj Close"].rolling(5).mean()
        df_copy["MA_10"] = df_copy["Adj Close"].rolling(10).mean()
        
        df_copy = df_copy.dropna()
        print(pd.DataFrame(future_predictions))

def next_day(df, model, threshold=0.45):
    
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Model is not trained yet.")
    
    latest = df.iloc[-1:]
    missing = [col for col in model_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features in input data: {missing}")
    
    X = latest[model_features]
    
    prob = model.predict_proba(X)[:, 1][0]
    pred = 1 if prob > threshold else 0
    
    print("Next Day Direction:", "UP" if pred == 1 else "DOWN")
    print("Confidence:", round(prob, 4))
    
    return pred, prob


next_day(df, model)
# Plot Date vs Close
plt.figure(figsize=(10, 5))
plt.plot(dp["Date"], dp["Close"])
plt.title("TCS Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()





