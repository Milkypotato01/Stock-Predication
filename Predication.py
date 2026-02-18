import yfinance as yf
import joblib
import pandas as pd
import numpy as np

from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing
from Features import Feature_data
import matplotlib.pyplot as plt

df = yf.download("TCS.NS", start="2024-01-01", end="2024-12-23") 
model = joblib.load("stock_model_v2.pkl") 
model_features = model.feature_names_in_ 
validate_stock_data(df) 
df = proper_preprocessing(df) 
df_raw = df.copy() #DF with Close Volumn etc df = Feature_data(df) print(df) print("Model Features:", model_features) future_predictions = []

def predict_future(df_raw, model, model_features =  model.feature_names_in_, days=4, threshold=0.47):

    df_price = df_raw.copy()
    predictions = []

    # Initial full feature calculation (only once)
    df_processed = proper_preprocessing(df_price)
    df_features = Feature_data(df_processed)

    # Define rolling window size (adjust if your indicators need more)
    window_size = 120  

    for i in range(days):

        last_row = df_features.iloc[-1:]
        X = last_row[model_features]

        prob = model.predict_proba(X)[0, 1]
        pred = int(prob > threshold)

        predictions.append({
            "Day": i+1,
            "Prediction": pred,
            "Probability": round(prob, 4)
        })

        # simulate return
        mean_up = df_features[df_features["Return"] > 0]["Return"].mean()
        mean_down = df_features[df_features["Return"] < 0]["Return"].mean()

        simulated_return = prob * mean_up + (1 - prob) * mean_down
        last_close = df_price["Close"].iloc[-1]
        new_close = last_close * (1 + simulated_return)

        new_row = df_price.iloc[-1:].copy()
        new_row["Close"] = new_close

        df_price = pd.concat([df_price, new_row], ignore_index=True)

        # 🔥 Recompute features ONLY on recent window
        df_processed = proper_preprocessing(df_price.tail(window_size))
        df_new_features = Feature_data(df_processed)

        # Append only the last newly computed feature row
        df_features = pd.concat(
            [df_features, df_new_features.iloc[-1:]],
            ignore_index=True
        )

    return pd.DataFrame(predictions)


# def predict_future(df_raw, model, model_features, days=4, threshold=0.47):

#     df_price = df_raw.copy()
#     predictions = []

#     for i in range(days):

#         df_processed = proper_preprocessing(df_price)
#         df_features = Feature_data(df_processed)

#         last_row = df_features.iloc[-1:]
#         X = last_row[model_features]

#         prob = model.predict_proba(X)[0, 1]
#         pred = int(prob > threshold)

#         predictions.append({
#             "Day": i+1,
#             "Prediction": pred,
#             "Probability": round(prob, 4)
#         })

#         # simulate return
#         mean_up = df_features[df_features["Return"] > 0]["Return"].mean()
#         mean_down = df_features[df_features["Return"] < 0]["Return"].mean()

#         simulated_return = prob * mean_up + (1 - prob) * mean_down
#         last_close = df_price["Close"].iloc[-1]
#         new_close = last_close * (1 + simulated_return)

#         new_row = df_price.iloc[-1:].copy()
#         new_row["Close"] = new_close

#         df_price = pd.concat([df_price, new_row], ignore_index=True)

#     return pd.DataFrame(predictions)


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

def plot_graph(df_raw):
    plt.figure(figsize=(10, 5))
    plt.plot(df_copy["Date"], df_copy["Close"])
    plt.title("TCS Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print(predict_future(df_raw, model, days=50))





