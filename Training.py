from Preprocessing import proper_preprocessing
from Features import test_train_divide


df = proper_preprocessing(r"C:\Users\dell\OneDrive\Desktop\Stock predictation Project\test data\indexProcessed.csv")
X_train, X_test, y_train, y_test = test_train_divide(df)

print(X_test)