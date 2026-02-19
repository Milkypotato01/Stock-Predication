import pandas as pd
import yfinance as yf
import joblib
from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing
from Features import Feature_data
from sklearn.preprocessing import LabelEncoder


def test_train_divide( tickers = [
    "TCS.NS",
    "RELIANCE.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "ITC.NS",
    "HINDUNILVR.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
    "ONGC.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "JSWSTEEL.NS",
    "COALINDIA.NS",
    "BPCL.NS",
    "HEROMOTOCO.NS",
    "BRITANNIA.NS",
    "DRREDDY.NS",
    "CIPLA.NS",
    "TECHM.NS",
    "INDUSINDBK.NS",
    "GRASIM.NS",
    "DIVISLAB.NS",
    "EICHERMOT.NS",
    "APOLLOHOSP.NS"
]
):

   
    encoder = LabelEncoder() 
    encoder.fit(tickers)
    joblib.dump(encoder, "stock_encoder.pkl")
    print("Label encoder saved as 'stock_encoder.pkl'")

    all_train = []
    all_test = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start="1970-01-01" , end="2024-12-23")
        
            validate_stock_data(df)
            df = proper_preprocessing(df)
            print(f"Preprocessing for {ticker} done")
            df["Stock"] = ticker
           
            df = Feature_data(df)
            df = df.sort_values("Date")
            
            split_index = int(len(df) * 0.8)
              
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()
    
            train_df["Stock_encoded"] = encoder.transform(train_df["Stock"])
            test_df["Stock_encoded"]  = encoder.transform(test_df["Stock"])
    
            all_train.append(train_df)
            all_test.append(test_df)
    
        except Exception as e:
                  print(f"Error downloading data for {ticker}: {e}")
                  continue
    final_train = pd.concat(all_train)
    final_test = pd.concat(all_test)
    
    X_train = final_train.drop(columns=["Date", "Target", "Stock"])
    y_train = final_train["Target"]
    
    X_test = final_test.drop(columns=["Date", "Target", "Stock"])
    y_test = final_test["Target"]

    print("Data splitting done")

    split_data = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test
    }
    
    print(X_train.columns)
    

    return split_data

# if __name__ == "__main__":
#     df = pd.read_pickle("processed_stock_data.pkl")
#     test_train_divide(df)
if __name__ == "__Evaulation__":
    test_train_divide()

