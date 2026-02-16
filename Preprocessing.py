import pandas as pd
import numpy as np

def proper_preprocessing(Data):
    #sorting 
    df = pd.read_csv(Data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    #readjusting
    df = df.drop(columns=["Index"])
    df = df.drop(columns=["Volume"])
    df = df.reset_index(drop=True)
    df = df[["Date", "Adj Close"]]
    
    # 1New Columne
    df["Return"] = df["Adj Close"].pct_change()
    df["MA_5"] = df["Adj Close"].rolling(window=5).mean()
    df["MA_10"] = df["Adj Close"].rolling(window=10).mean()
    df["MA_20"] = df["Adj Close"].rolling(window=20).mean()
    df["Volatility_5"] = df["Return"].rolling(window=5).std()
    df["Volatility_10"] = df["Return"].rolling(window=10).std()
    df["Target"] = np.where(df["Adj Close"].shift(-1) > df["Adj Close"], 1, 0)
    
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
