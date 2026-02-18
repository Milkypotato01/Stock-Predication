import pandas as pd
import numpy as np
import yfinance as yf
from Input_file_Validating import validate_stock_data

df = yf.download("RELIANCE.NS", start="1970-01-01", end="2024-12-31")
validate_stock_data(df)
def proper_preprocessing(df):
    #sorting 
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    
    df.dropna(subset=["Date", "Close", "Volume"], inplace=True)

    df = df.sort_values("Date")
    
    #readjusting
    df = df.drop(columns=["Index"], errors="ignore")
    df = df.reset_index(drop=True)
    df = df[["Date", "Close" , "Volume"]]
    print("Preprocessing Done")
    return df

print(proper_preprocessing(df))
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
