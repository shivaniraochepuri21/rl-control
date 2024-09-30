import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Assuming 'data' is your DataFrame from the previous step

# 1. Data Preparation
X = data[['K', 'λ1', 'λ2', 'w1', 'w2']]
y_Tr = data['Tr (sec)']
y_SSE = data['SSE (%)']
y_CE = data['CE']

# 2. Exploratory Data Analysis (EDA)
sns.pairplot(data)
plt.show()

# 3. Correlation Analysis
corr_matrix = data.corr()
print(corr_matrix[['Tr (sec)', 'SSE (%)', 'CE']])

# 4. Multiple Linear Regression for Tr (sec)
X_train, X_test, y_train, y_test = train_test_split(X, y_Tr, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"R-squared for Tr: {r2_score(y_test, y_pred)}")
print(f"Coefficients: {model.coef_}")

# 5. Polynomial Regression for Tr (sec)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_Tr, test_size=0.2, random_state=42)

poly_model = LinearRegression()
poly_model.fit(X_train, y_train)
y_poly_pred = poly_model.predict(X_test)

print(f"R-squared for Polynomial Regression Tr: {r2_score(y_test, y_poly_pred)}")