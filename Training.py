import joblib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from Data_spliting import test_train_divide
from Input_file_Validating import validate_stock_data
from Preprocessing import proper_preprocessing
from Features import Feature_data


df = yf.download("RELIANCE.NS", start="1970-01-01", end="2024-12-31")
validate_stock_data(df)
df = proper_preprocessing(df)
df = Feature_data(df)
data = test_train_divide(df)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]


model = RandomForestClassifier(
    n_estimators=400,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=5,
    class_weight='balanced',
    max_features="sqrt",
    random_state=42
)

print("Training the model...")

model.fit(X_train, y_train)
joblib.dump(model, "stock_model_v5.pkl")

print("Model trained and saved as stock_model.pkl")


