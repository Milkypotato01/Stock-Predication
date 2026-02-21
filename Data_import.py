import joblib

# model_v2 = joblib.load("stock_model_v2.pkl") #55-60%
# model_v3 = joblib.load("stock_model_v3.pkl") # 52%
model_v4 = joblib.load("stock_model_v4.pkl")
# # model_v5 = joblib.load("stock_model_v5.pkl")

# model_v2_features = model_v2.feature_names_in_.tolist()
# model_v3_features = model_v3.feature_names_in_.tolist()
model_v4_features = model_v4.feature_names_in_.tolist()
# # model_v5_features = model_v5.feature_names_in_.tolist()

# print("Model v2 features:", model_v2_features)
# print("Model v3 features:", model_v3_features)
print("Model v4 features:", model_v4_features)
# # print("Model v5 features:", model_v5_features)

