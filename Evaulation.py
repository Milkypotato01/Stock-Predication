
from xml.parsers.expat import model
from Preprocessing import proper_preprocessing
from Data_spliting import test_train_divide
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

model = joblib.load("stock_model.pkl")

df = proper_preprocessing(r"C:\Users\dell\OneDrive\Desktop\Stock predictation Project\test data\indexProcessed.csv")
X_train, X_test, y_train, y_test = test_train_divide(df)

print("Training the model...")

model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
y_pred = (y_prob > 0.45).astype(int)   # try 0.45 instead of 0.5


def model_evaluation(y_test, y_pred, y_prob):
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    feature_importance = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)
    
    print(feature_importance)
    
    
    baseline = [1] * len(y_test)
    print("Baseline Accuracy:", accuracy_score(y_test, baseline))
    
    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

print("Evaluating the model...")
model_evaluation(y_test, y_pred, y_prob)

