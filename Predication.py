from Preprocessing import proper_preprocessing
import joblib
import pandas as pd

model = joblib.load("stock_model.pkl")
model_features = model.feature_names_in_
df = proper_preprocessing()

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

def next_day():
    latest = df.iloc[-1:]
    X = latest[model_features]
    
    prob = model.predict_proba(X)[:, 1][0]
    pred = 1 if prob > 0.45 else 0
    
    print("Next Day Direction:", "UP" if pred == 1 else "DOWN")
    print("Confidence:", prob)

