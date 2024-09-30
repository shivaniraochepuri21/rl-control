import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Given data
data = {
    "K": [4, 0.1, 1, 4, 4, 0.01, 0.01],
    "λ1": [0.0, 0.0, 0.0, 0.8, 0.8, 3.0, 2.0],
    "λ2": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
    "w1": [1, 1, 1, 1, 0.5, 1, 1],
    "w2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "Tr (sec)": [0.55, 23.4, 2.2, 0.55, 0.55, 0.4, 2.1],
    "SSE (%)": [0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 5.0],
    "CE": [0.79, 39.68, 1.03, 0.8, 0.77, 1.71, 3.17]
}

df = pd.DataFrame(data)

# Select the data point with K = 4 to be the test point
test_point_index = df[df["K"] == 4].index[0]
test_point = df.loc[test_point_index]
df_train = df.drop(test_point_index)

# Separate features and target
X_train = df_train[["K", "λ1", "λ2", "w1", "w2"]]
y_train_tr = df_train["Tr (sec)"]
y_train_sse = df_train["SSE (%)"]
y_train_ce = df_train["CE"]

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)

# Train models
model_tr = LinearRegression().fit(X_train_poly, y_train_tr)
model_sse = LinearRegression().fit(X_train_poly, y_train_sse)
model_ce = LinearRegression().fit(X_train_poly, y_train_ce)

# Prepare the test point
X_test = test_point[["K", "λ1", "λ2", "w1", "w2"]].values.reshape(1, -1)
X_test_poly = poly.transform(X_test)

# Predict
tr_pred = model_tr.predict(X_test_poly)
sse_pred = model_sse.predict(X_test_poly)
ce_pred = model_ce.predict(X_test_poly)

# Actual values
test_point_actual = test_point[["Tr (sec)", "SSE (%)", "CE"]].values

# Print results
print(f"Actual Tr: {test_point_actual[0]}")
print(f"Predicted Tr: {tr_pred[0]}")

print(f"Actual SSE: {test_point_actual[1]}")
print(f"Predicted SSE: {sse_pred[0]}")

print(f"Actual CE: {test_point_actual[2]}")
print(f"Predicted CE: {ce_pred[0]}")
