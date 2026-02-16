import pandas as pd
import numpy as np

def proper_preprocessing(Data):
    #sorting 
    df = pd.read_csv(Data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    #readjusting
    df = df.drop(columns=["Index", "Volume"])
    df = df.reset_index(drop=True)
    df = df[["Date", "Adj Close"]]
    
    # Derieving features
    df["Return"] = df["Adj Close"].pct_change()

    df["MA_5"] = df["Adj Close"].rolling(window=5).mean()
    df["MA_10"] = df["Adj Close"].rolling(window=10).mean()
    df["MA_20"] = df["Adj Close"].rolling(window=20).mean()

    df["Volatility_5"] = df["Return"].rolling(window=5).std()
    df["Volatility_10"] = df["Return"].rolling(window=10).std()

    df["Target"] = np.where(df["Adj Close"].shift(-1) > df["Adj Close"], 1, 0)

    # RSI (14)
    delta = df["Adj Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["Adj Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Adj Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag2"] = df["Return"].shift(2)

    df = df.drop(columns=["Adj Close"])
 
    # Remove NaN 
    df = df.dropna()
    df = df.reset_index(drop=True)
    print("Preprocessing Done")
    df.to_pickle("processed_stock_data.pkl")
    print("Data saved as processed_stock_data.pkl")
    
    return df

proper_preprocessing(r"C:\Users\dell\OneDrive\Desktop\Stock predictation Project\test data\indexProcessed.csv")

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
