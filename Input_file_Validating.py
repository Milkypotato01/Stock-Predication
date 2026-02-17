import pandas as pd
import yfinance as yf

df = yf.download("RELIANCE.NS", start="1970-01-01", end="2024-12-31")

def validate_stock_data(df):

    if df is None or df.empty:
        print("DataFrame is empty.")
        return False

    df.reset_index(inplace=True)
    df.columns = df.columns.get_level_values(0)
    df.columns.name = None
    
    required_columns = ["Date", "Close", "Volume"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"Missing required columns: {missing_columns}")
        return False

    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        print("Column 'Date' is not in datetime format.")
        return False

    if not pd.api.types.is_numeric_dtype(df["Close"]):
        print("Column 'Close' is not numeric.")
        return False

    if not pd.api.types.is_numeric_dtype(df["Volume"]):
        print("Column 'Volume' is not numeric.")
        return False

    print("Validation successful.")
    return True
