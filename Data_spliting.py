import joblib
import pandas as pd

def test_train_divide(df):
    X = df.drop(columns=["Date", "Adj Close", "Target"])
    y = df["Target"]
    
    # Calculate split index (80%)
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Data splitting done")

    joblib.dump(X_train, "X_train.pkl")
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_train, "y_train.pkl")
    joblib.dump(y_test, "y_test.pkl")

    print("Data saved to disk")

test_train_divide(pd.read_pickle("processed_stock_data.pkl"))
    
