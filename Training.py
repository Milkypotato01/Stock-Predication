import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_pickle("processed_stock_data.pkl")    

X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

print("Training the model...")

model.fit(X_train, y_train)
joblib.dump(model, "stock_model.pkl")

print("Model trained and saved as stock_model.pkl")


