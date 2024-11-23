# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import streamlit as st

# Load data
def load_data():
    return pd.read_csv('listings.csv')

bnb = load_data()

# Display the first few rows of the dataframe
st.write(bnb.head())

# Data Cleaning Function: Check for missing or infinite values
def clean_data(X, y):
    # Remove rows where there are any missing values in either X or y
    X_clean = X[~X.isnull().any(axis=1)]
    y_clean = y[X_clean.index]  # Ensure the y data corresponds to the cleaned X

    # Remove any rows with infinite values
    X_clean = X_clean[~X_clean.isin([np.inf, -np.inf]).any(axis=1)]
    y_clean = y_clean[X_clean.index]

    return X_clean, y_clean

# Display data info
st.write(bnb.info())

# Clean data
X = bnb.drop('neighbourhood_group', axis=1)  # Drop target variable
y = bnb['neighbourhood_group']  # Target variable

# Clean the data
X_clean, y_clean = clean_data(X, y)

# Train-test split after cleaning
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.30, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Display coefficients of the linear regression model
st.write('Linear Regression Coefficients:', lr.coef_)

# Predictions
y_pred = lr.predict(X_test_scaled)

# Display predictions and true values
st.write('Predicted values:', y_pred[:5])
st.write('True values:', y_test.head())

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(bnb['neighbourhood_group'], color="skyblue", ax=axes[0])
sns.histplot(bnb['room_type'], color="olive", ax=axes[1])
plt.show()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(bnb.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
plt.show()

# Feature importance using a simple decision tree model
from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree model
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_scaled, y_train)

# Display feature importances
st.write('Feature Importances (Decision Tree):', dt.feature_importances_)

# Model evaluation
from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the Linear Regression model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r2}')

# Visualizing predictions vs. true values
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax)
ax.set_title('True vs Predicted values')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
plt.show()
