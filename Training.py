import joblib
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from Data_spliting import test_train_divide


data = test_train_divide()

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
joblib.dump(model, "stock_model_v3.pkl")

print("Model trained and saved as stock_model_v3.pkl")


