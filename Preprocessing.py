import pandas as pd
import numpy as np
import yfinance as yf
from Input_file_Validating import validate_stock_data

df = yf.download("RELIANCE.NS", start="1970-01-01", end="2024-12-31")
validate_stock_data(df)

def proper_preprocessing(df):

    # 1. Check if empty
    if df is None or df.empty:
        print("Preprocessing skipped: Empty DataFrame")
        return pd.DataFrame()

    # 2. Fix MultiIndex columns (yfinance issue)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 3. Ensure Date column exists
    if "Date" not in df.columns:
        if df.index.name == "Date" or isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        else:
            print("Preprocessing skipped: No Date column")
            return pd.DataFrame()

    # 4. Ensure required columns exist
    required_cols = ["Date", "Close", "Volume", "High", "Low"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        print(f"Preprocessing skipped: Missing columns {missing}")
        return pd.DataFrame()

    # 5. Convert types safely
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

    # 6. Drop invalid rows
    df.dropna(subset=["Date", "Close", "Volume"], inplace=True)

    # 7. Sort properly
    df = df.sort_values("Date")

    # 8. Remove unwanted columns safely
    df = df.drop(columns=["Index"], errors="ignore")

    # 9. Reset index
    df = df.reset_index(drop=True)

    # 10. Keep only required columns
    df = df[["Date", "High", "Low", "Close", "Volume"]]

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
