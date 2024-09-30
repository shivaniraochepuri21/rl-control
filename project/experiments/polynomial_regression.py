import numpy as np
import pandas as pd

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
X = df[["K", "λ1", "λ2", "w1", "w2"]]
y_tr = df["Tr (sec)"]
y_sse = df["SSE (%)"]
y_ce = df["CE"]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Create and fit models
model_tr = LinearRegression().fit(X_poly, y_tr)
model_sse = LinearRegression().fit(X_poly, y_sse)
model_ce = LinearRegression().fit(X_poly, y_ce)

# Predict and evaluate
y_tr_pred = model_tr.predict(X_poly)
y_sse_pred = model_sse.predict(X_poly)
y_ce_pred = model_ce.predict(X_poly)

# Evaluation
print(f"Tr Model MSE: {mean_squared_error(y_tr, y_tr_pred)}")
print(f"SSE Model MSE: {mean_squared_error(y_sse, y_sse_pred)}")
print(f"CE Model MSE: {mean_squared_error(y_ce, y_ce_pred)}")

# To predict new values
new_params = np.array([[2, 1.0, 0.1, 0.7, 0.002]])  # Example new parameters
new_params_poly = poly.transform(new_params)

tr_pred = model_tr.predict(new_params_poly)
sse_pred = model_sse.predict(new_params_poly)
ce_pred = model_ce.predict(new_params_poly)

print(f"Predicted Tr: {tr_pred}")
print(f"Predicted SSE: {sse_pred}")
print(f"Predicted CE: {ce_pred}")
