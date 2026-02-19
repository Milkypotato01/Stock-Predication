import pandas as pd
import yfinance as yf
import joblib
from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing
from Features import Feature_data



def test_train_divide(tickers = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","LT.NS","ITC.NS","HINDUNILVR.NS","AXISBANK.NS",
    "KOTAKBANK.NS","BAJFINANCE.NS","BAJAJFINSV.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","ULTRACEMCO.NS","TITAN.NS","WIPRO.NS","ONGC.NS",
    "NTPC.NS","POWERGRID.NS","ADANIENT.NS","ADANIPORTS.NS","TATAMOTORS.NS",
    "TATASTEEL.NS","JSWSTEEL.NS","COALINDIA.NS","BPCL.NS","HEROMOTOCO.NS",
    "BRITANNIA.NS","DRREDDY.NS","CIPLA.NS","TECHM.NS","INDUSINDBK.NS",
    "GRASIM.NS","DIVISLAB.NS","EICHERMOT.NS","APOLLOHOSP.NS","UPL.NS",

    "HCLTECH.NS","DABUR.NS","GODREJCP.NS","PIDILITIND.NS","BERGEPAINT.NS",
    "SIEMENS.NS","ABB.NS","HAVELLS.NS","BOSCHLTD.NS","MCDOWELL-N.NS",
    "COLPAL.NS","ICICIPRULI.NS","ICICIGI.NS","BAJAJ-AUTO.NS","TVSMOTOR.NS",
    "MOTHERSUMI.NS","ASHOKLEY.NS","ESCORTS.NS","MRF.NS","CEAT.NS",

    "AMBUJACEM.NS","ACC.NS","SHREECEM.NS","DALBHARAT.NS","RAMCOCEM.NS",
    "JINDALSTEL.NS","SAIL.NS","NMDC.NS","VEDL.NS","HINDALCO.NS",

    "BANKBARODA.NS","PNB.NS","CANBK.NS","UNIONBANK.NS","IDFCFIRSTB.NS",
    "FEDERALBNK.NS","RBLBANK.NS","BANDHANBNK.NS","YESBANK.NS","AUROPHARMA.NS",

    "LUPIN.NS","ALKEM.NS","BIOCON.NS","TORNTPHARM.NS","ZYDUSLIFE.NS",
    "GLENMARK.NS","ABBOTINDIA.NS","SANOFI.NS","PFIZER.NS","IPCALAB.NS",

    "ADANIGREEN.NS","ADANITRANS.NS","ADANIPOWER.NS","ATGL.NS","GAIL.NS",
    "IOC.NS","PETRONET.NS","IGL.NS","MGL.NS","GSPL.NS",

    "TATAPOWER.NS","TORNTPOWER.NS","CESC.NS","NHPC.NS","SJVN.NS",
    "IRCTC.NS","IRFC.NS","RVNL.NS","CONCOR.NS","BHEL.NS",

    "BEL.NS","HAL.NS","BDL.NS","COCHINSHIP.NS","MAZDOCK.NS",
    "GRSE.NS","LTTS.NS","MPHASIS.NS","PERSISTENT.NS","COFORGE.NS",

    "LTIM.NS","OFSS.NS","KPITTECH.NS","ZENSARTECH.NS","CYIENT.NS",
    "TANLA.NS","SONATSOFTW.NS","HAPPSTMNDS.NS","ROUTE.NS","NEWGEN.NS",

    "DMART.NS","TRENT.NS","VBL.NS","UNITDSPR.NS","UBL.NS",
    "RADICO.NS","EMAMILTD.NS","MARICO.NS","PATANJALI.NS","JUBLFOOD.NS",

    "WESTLIFE.NS","DEVYANI.NS","TATACONSUM.NS","NESTLEIND.NS","HINDZINC.NS",
    "JSWENERGY.NS","SUZLON.NS","INOXWIND.NS","BORORENEW.NS","POLYCAB.NS",

    "KEI.NS","FINCABLES.NS","RRKABEL.NS","APLAPOLLO.NS","ASTRAL.NS",
    "SUPREMEIND.NS","TIMKEN.NS","SKFINDIA.NS","SCHAEFFLER.NS","CUMMINSIND.NS",

    "THERMAX.NS","AIAENG.NS","ISGEC.NS","KSB.NS","ELGIEQUIP.NS",
    "VOLTAS.NS","BLUESTARCO.NS","WHIRLPOOL.NS","HONAUT.NS","3MINDIA.NS",

    "PAGEIND.NS","RELAXO.NS","BATAINDIA.NS","METROPOLIS.NS","LALPATHLAB.NS",
    "DRLALCHANDANI.NS","KIMS.NS","RAINBOW.NS","FORTIS.NS","MAXHEALTH.NS",

    "INDIGO.NS","SPICEJET.NS","DELHIVERY.NS","TCI.NS","VRLLOG.NS",
    "ALLCARGO.NS","GMRINFRA.NS","GVKPIL.NS","NBCC.NS","HFCL.NS",

    "IDEA.NS","BHARTIARTL.NS","TATACOMM.NS","RAILTEL.NS","MTNL.NS",

    "PEL.NS","PFC.NS","RECLTD.NS","MUTHOOTFIN.NS","CHOLAFIN.NS",
    "SHRIRAMFIN.NS","LICHSGFIN.NS","HUDCO.NS","IIFL.NS","JMFINANCIL.NS",

    "CROMPTON.NS","ORIENTELEC.NS","VGUARD.NS","BAJAJELEC.NS","SYMPHONY.NS"
]

):

    all_train = []
    all_test = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, start="1970-01-01" , end="2024-12-23")
        
            validate_stock_data(df)
            df = proper_preprocessing(df)
            print(f"Preprocessing for {ticker} done")
           
            df = Feature_data(df)
            df = df.sort_values("Date")
            
            split_index = int(len(df) * 0.8)
              
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()
    
    
            all_train.append(train_df)
            all_test.append(test_df)
    
        except Exception as e:
                  print(f"Error downloading data for {ticker}: {e}")
                  continue
    final_train = pd.concat(all_train)
    final_test = pd.concat(all_test)
    
    X_train = final_train.drop(columns=["Date", "Target"])
    y_train = final_train["Target"]
    
    X_test = final_test.drop(columns=["Date", "Target"])
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

