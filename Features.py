import numpy as np
import joblib
import yfinance as yf

from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing

df = yf.download("RELIANCE.NS", start="1970-01-01", end="2024-12-31")
validate_stock_data(df)
df = proper_preprocessing(df)

def Feature_data(df):
    
    # Derieving features
    df["Volume_MA_5"] = df["Volume"].rolling(5).mean()
    df["Volume_Spike"] = df["Volume"] / df["Volume_MA_5"]
    df.drop(columns=["Volume_MA_5"], inplace=True)
    df["OBV"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["VWAP_Distance"] = df["Close"] - df["VWAP"]
    df["Volume_Zscore_20"] = (
        (df["Volume"] - df["Volume"].rolling(20).mean()) /
        df["Volume"].rolling(20).std()
    )

# df["Price_Volume_Corr"] = df["Return"].rolling(10).corr(df["Volume"])

    df["Return"] = df["Close"].pct_change()
    # df["MA_5"] = df["Close"].rolling(window=5).mean()
    # df["MA_10"] = df["Close"].rolling(window=10).mean()
    # df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["Volatility_5"] = df["Return"].rolling(window=5).std()
    df["Volatility_10"] = df["Return"].rolling(window=10).std()
    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

    # RSI (14)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)

    df = df.drop(columns=["Close"])
 
    # Remove NaN 
    df = df.dropna()
    df = df.reset_index(drop=True)
    print("Preprocessing [Feature part (2/2)] Done")
    # df.to_pickle("processed_stock_data.pkl")
    print("Data saved as processed_stock_data.pkl")
    
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
