import pandas as pd
import yfinance as yf
import joblib
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from Data_spliting import test_train_divide
from sklearn.metrics import precision_recall_curve

# df = yf.download("TCS.NS", start="1970-01-01", end="2024-12-31")

data = test_train_divide()

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Loading the model...")
model = joblib.load("stock_model_v6.pkl")
model_features = model.feature_names_in_.tolist()
print("Model Features:", model_features)

print("Evaluating the model...")

y_prob = model.predict_proba(X_test)[:, 1] 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print("Optimal threshold:", optimal_threshold)

y_pred = (y_prob >= optimal_threshold).astype(int)


print( "Y_prob_mean" , y_prob.mean())
print( " Y_prob_describe" ,pd.Series(y_prob).describe())


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

    strategy_return = df_results["Predicted"] * X_train["Return"].iloc[-len(y_test):].values
    print("Strategy Mean Return:", strategy_return.mean() , "%")

    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

    Model_report = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred),
        "Feature Importance": feature_importance,
        "Baseline Accuracy": accuracy_score(y_test, baseline),
        "Train Accuracy": model.score(X_train, y_train),
        "Test Accuracy": model.score(X_test, y_test),
        "Strategy Mean Return": strategy_return.mean()
    }

    print("Precision:", precision)
    print("Recall:", recall)

    
    model_features = X_train.columns.tolist()
    joblib.dump(Model_report, "model_Evaluation_report.pkl")
    print("feature of model" , model_features)


print("Evaluating the model...")
model_evaluation(y_test, y_pred, y_prob)

