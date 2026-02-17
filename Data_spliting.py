import joblib
import pandas as pd

def test_train_divide(df):
    X = df.drop(columns=["Date", "Target"])
    y = df["Target"]
    
    # Calculate split index (80%)
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("Data splitting done")

    split_data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
    }

    return split_data

# if __name__ == "__main__":
#     df = pd.read_pickle("processed_stock_data.pkl")
#     test_train_divide(df)
