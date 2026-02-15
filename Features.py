def test_train_divide(df):
    X = df.drop(columns=["Date", "Adj Close", "Target"])
    y = df["Target"]
    
    # Calculate split index (80%)
    split_index = int(len(df) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    return X_train, X_test, y_train, y_test