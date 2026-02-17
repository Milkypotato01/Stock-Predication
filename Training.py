from Data_spliting import test_train_divide
from sklearn.ensemble import RandomForestClassifier
import joblib

data = joblib.load("processed_stock_data.pkl")
data = test_train_divide(data)
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]


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


