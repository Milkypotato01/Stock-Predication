import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

df = pd.read_pickle("processed_stock_data.pkl")

print("Loading the model...")
model = joblib.load("stock_model.pkl")
data = joblib.load("data_split.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Evaluating the model...")

y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
y_pred = (y_prob > 0.45).astype(int)   # try 0.45 instead of 0.5


def model_evaluation(y_test, y_pred, y_prob):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    ).sort_values(ascending=False)
    
    print(feature_importance)
    
    baseline = [y_train.mode()[0]] * len(y_test)
    print("Baseline Accuracy:", accuracy_score(y_test, baseline))
    
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

    df_results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
    })

    strategy_return = df_results["Predicted"] * df["Return"].iloc[-len(y_test):].values
    print("Strategy Mean Return:", strategy_return.mean() , "%")
    
    model_features = X_train.columns.tolist()
    joblib.dump(model_features, "model_features.pkl")



print("Evaluating the model...")
model_evaluation(y_test, y_pred, y_prob)

