from sklearn.ensemble import RandomForestClassifier
from Preprocessing import proper_preprocessing
from Data_spliting import test_train_divide
import joblib

df = proper_preprocessing(r"C:\Users\dell\OneDrive\Desktop\Stock predictation Project\test data\indexProcessed.csv")
X_train, X_test, y_train, y_test = test_train_divide(df)

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

print("Model saved as stock_model.pkl")

