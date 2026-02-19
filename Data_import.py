import joblib

model_v2 = joblib.load("stock_model_v2.pkl")
model_v3 = joblib.load("stock_model_v3.pkl")

model_v2_features = model_v2.feature_names_in_.tolist()
model_v3_features = model_v3.feature_names_in_.tolist()

print("Model v2 features:", model_v2_features)
print("Model v5 features:", model_v3_features)