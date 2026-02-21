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
    n_estimators=800,
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=42
)


print("Training the model...")

model.fit(X_train, y_train)
joblib.dump(model, "stock_model_v6.pkl")

print("Model trained and saved as stock_model_v6.pkl")


