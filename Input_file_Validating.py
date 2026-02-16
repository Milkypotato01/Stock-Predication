import joblib
import pandas as pd

def validate_input_file(file_path):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print("File could not be read.")
        print("Error:", e)
        return False

    if df.empty:
        print("File is empty.")
        return False

    required_columns = ["Date", "Adj Close"]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print("Missing required columns:", missing_columns)
        return False

    # Validate Date format
    try:
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        print("Date column is not in valid datetime format.")
        return False

    # Validate Adj Close numeric
    if not pd.api.types.is_numeric_dtype(df["Adj Close"]):
        print("Adj Close column is not numeric.")
        return False

    print("Validation passed. File structure is correct.")
    return True

file_path = r"C:\Users\dell\OneDrive\Desktop\Stock predictation Project\test data\indexProcessed.csv"

if validate_input_file(file_path):
    print("Ready for preprocessing...")
    print("All required model features present.")
else:
    print("Fix the file before Preprocessing.")

