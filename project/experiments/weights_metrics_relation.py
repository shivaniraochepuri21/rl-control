# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from sklearn.linear_model import LinearRegression
# # from sklearn.preprocessing import PolynomialFeatures
# # from sklearn.model_selection import train_test_split
# # from sklearn.metrics import mean_squared_error, r2_score

# # # Define the data
# # data = {
# #     "S.No.": [1, 2, 3, 4, 5, 6, 7],
# #     "K": [4, 0.1, 1, 4, 4, 0.01, 0.01],
# #     "λ1": [0.0, 0.0, 0.0, 0.8, 0.8, 3.0, 2.0],
# #     "λ2": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
# #     "w1": [1, 1, 1, 1, 0.5, 1, 1],
# #     "w2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
# #     "Tr (sec)": [0.55, 23.4, 2.2, 0.55, 0.55, 0.4, 2.1],
# #     "SSE (%)": [0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 5.0],
# #     "CE": [0.79, 39.68, 1.03, 0.8, 0.77, 1.71, 3.17]
# # }

# # # Convert to DataFrame
# # df = pd.DataFrame(data)

# # # Display the DataFrame
# # print(df)

# # # 1. Data Preparation
# # X = df[['K', 'λ1', 'λ2', 'w1', 'w2']]  # Use 'df' instead of 'data'
# # y_Tr = df['Tr (sec)']  # Use 'df' instead of 'data'
# # y_SSE = df['SSE (%)']  # Use 'df' instead of 'data'
# # y_CE = df['CE']  # Use 'df' instead of 'data'

# # # 2. Exploratory Data Analysis (EDA)
# # sns.pairplot(df)
# # plt.show()

# # # 3. Correlation Analysis
# # corr_matrix = df.corr()
# # print(corr_matrix[['Tr (sec)', 'SSE (%)', 'CE']])

# # # 4. Multiple Linear Regression for Tr (sec)
# # X_train, X_test, y_train, y_test = train_test_split(X, y_Tr, test_size=0.2, random_state=42)
# # model = LinearRegression()
# # model.fit(X_train, y_train)
# # y_pred = model.predict(X_test)

# # print(f"R-squared for Tr: {r2_score(y_test, y_pred)}")
# # print(f"Coefficients: {model.coef_}")

# # # 5. Polynomial Regression for Tr (sec)
# # poly = PolynomialFeatures(degree=2)
# # X_poly = poly.fit_transform(X)
# # X_train, X_test, y_train, y_test = train_test_split(X_poly, y_Tr, test_size=0.2, random_state=42)

# # poly_model = LinearRegression()
# # poly_model.fit(X_train, y_train)
# # y_poly_pred = poly_model.predict(X_test)

# # print(f"R-squared for Polynomial Regression Tr: {r2_score(y_test, y_poly_pred)}")

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Define the data
# data = {
#     # "S.No.": [1, 2, 3, 4, 5, 6, 7],
#     "K": [4, 0.1, 1, 4, 4, 0.01, 0.01],
#     "λ1": [0.0, 0.0, 0.0, 0.8, 0.8, 3.0, 2.0],
#     "λ2": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
#     "w1": [1, 1, 1, 1, 0.5, 1, 1],
#     "w2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
#     "Tr (sec)": [0.55, 23.4, 2.2, 0.55, 0.55, 0.4, 2.1],
#     "SSE (%)": [0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 5.0],
#     "CE": [0.79, 39.68, 1.03, 0.8, 0.77, 1.71, 3.17]
# }

# # Convert to DataFrame
# df = pd.DataFrame(data)

# # Setting up the figure
# fig, axs = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=False)
# fig.suptitle('Performance Metrics vs Control Parameters', fontsize=16)

# # Defining parameter names and performance metrics
# params = ['K', 'λ1', 'λ2', 'w1', 'w2']
# metrics = ['Tr (sec)', 'SSE (%)', 'CE']

# # Plotting each metric against each parameter
# for i, metric in enumerate(metrics):
#     for j, param in enumerate(params):
#         sns.regplot(x=param, y=metric, data=df, ax=axs[i, j], ci=None)
#         axs[i, j].set_xlim(0, max(df[param]) * 1.1)
#         axs[i, j].set_ylim(0, max(df[metric]) * 1.1)
#         axs[i, j].set_title(f'{metric} vs {param}')

# # Removing unused subplots (if any)
# for k in range(len(metrics) * len(params), 9):
#     fig.delaxes(axs.flatten()[k])

# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define the data
data = {
    "S.No.": [1, 2, 3, 4, 5, 6, 7],
    "K": [4, 0.1, 1, 4, 4, 0.01, 0.01],
    "λ1": [0.0, 0.0, 0.0, 0.8, 0.8, 3.0, 2.0],
    "λ2": [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1],
    "w1": [1, 1, 1, 1, 0.5, 1, 1],
    "w2": [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001],
    "Tr (sec)": [0.55, 23.4, 2.2, 0.55, 0.55, 0.4, 2.1],
    "SSE (%)": [0.0, 0.0, 0.0, 0.0, 0.2, 1.0, 5.0],
    "CE": [0.79, 39.68, 1.03, 0.8, 0.77, 1.71, 3.17]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define parameters and performance metrics
params = ['K', 'λ1', 'λ2', 'w1', 'w2']
metrics = ['Tr (sec)', 'SSE (%)', 'CE']

# Calculate the number of rows needed
n_cols = len(params)
n_rows = len(metrics)

# Setting up the figure with dynamic size based on the number of plots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=False, sharey=False)
fig.suptitle('Performance Metrics vs Control Parameters', fontsize=16)

# Plotting each metric against each parameter
for i, metric in enumerate(metrics):
    for j, param in enumerate(params):
        sns.regplot(x=param, y=metric, data=df, ax=axs[i, j], ci=None)
        axs[i, j].set_xlim(0, max(df[param]) * 1.1)
        axs[i, j].set_ylim(0, max(df[metric]) * 1.1)
        axs[i, j].set_title(f'{metric} vs {param}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
