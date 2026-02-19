import numpy as np
import joblib
import yfinance as yf

from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing

df = yf.download("RELIANCE.NS", start="1970-01-01", end="2024-12-31")
validate_stock_data(df)
df = proper_preprocessing(df)

def Feature_data(df):
    
        # RETURNS (core signal)
    df["Return"] = df["Close"].pct_change()
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)
    df["Return_Lag5"] = df["Return"].shift(5)
    
    # MOMENTUM
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["Momentum_10"] = df["Close"] / df["Close"].shift(10) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    
    # VOLATILITY
    df["Volatility_5"] = df["Return"].rolling(5).std()
    df["Volatility_10"] = df["Return"].rolling(10).std()
    df["Volatility_20"] = df["Return"].rolling(20).std()
    
    # TREND (normalized)
    ema50 = df["Close"].ewm(span=50).mean()
    ema200 = df["Close"].ewm(span=200).mean()
    
    df["Trend_Strength"] = (ema50 / ema200) - 1
    df["Price_vs_EMA50"] = (df["Close"] / ema50) - 1
    df["Price_vs_EMA200"] = (df["Close"] / ema200) - 1
    
    # VOLUME normalized
    # df["Volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    # df["Volume_z"] = (
    #     (df["Volume"] - df["Volume"].rolling(20).mean()) /
    #     df["Volume"].rolling(20).std()
    # )
    
    # RSI normalized
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    df["RSI"] = (100 - (100 / (1 + rs))) / 100
    
    df["Trend_acceleration"] = df["Momentum_5"] - df["Momentum_20"]

# EMA
    df["EMA_10"] = df["Close"].ewm(span=10).mean()
    df["EMA_20"] = df["Close"].ewm(span=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50).mean()
    
    # Normalized EMA features
    df["Price_vs_EMA10"] = df["Close"] / df["EMA_10"]
    df["Price_vs_EMA20"] = df["Close"] / df["EMA_20"]
    df["Price_vs_EMA50"] = df["Close"] / df["EMA_50"]
    
    # Trend signal

    
    # ATR
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_pct"] = df["ATR"] / df["Close"]
    
    # Breakout feature
    df["Breakout_20"] = df["Close"] / df["Close"].rolling(20).max()
    
    # Volume spike
    # df["Volume_spike"] = df["Volume"] / df["Volume"].rolling(20).mean()
    
    
    # DROP stock-dependent columns
    df.drop(columns=[
        "EMA_10",
        "EMA_20",
        "EMA_50",
        "ATR" ,
        "High",
        "Low"
    ], inplace=True, errors="ignore")
    
 
    df["Future_Return_5"] = df["Close"].shift(-5) / df["Close"] - 1
    df["Target"] = (df["Future_Return_5"] > 0.01).astype(int)


    df = df.drop(columns=["Close" , "Volume" , "Future_Return_5"] , errors="ignore")
 
    # Remove NaN 
    df = df.dropna()
    df = df.reset_index(drop=True)
    print("Preprocessing [Feature part (2/2)] Done")
    # df.to_pickle("processed_stock_data.pkl")
    
    return df



# print("\nZero volume rows:", len(zero_volume))
# # Check duplicate dates
# duplicate_dates = df[df.duplicated(subset=["Date"])]
# print("\nDuplicate date rows:", len(duplicate_dates))
# print(df["Date"].nunique())
# print(df[df["Date"] == df["Date"].iloc[0]])
# print(df.duplicated().sum())
# print(df.head(20))
# print(df.duplicated().sum())
# print(df["Date"].min(), df["Date"].max())
