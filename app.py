# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import streamlit as st
import warnings
from matplotlib.cbook import boxplot_stats
from sklearn.model_selection import train_test_split
from statsmodels.formula.api import ols
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Suppress warnings
warnings.filterwarnings("ignore")

# Load data
@st.cache_data
def load_data():
    bnb = pd.read_csv('listings.csv')
    return bnb

bnb = load_data()

# Display basic dataset information
st.title("Airbnb Price Prediction")
st.header("Dataset Overview")

st.write("Shape of the dataset:", bnb.shape)
st.write("Dataset Info:")
st.write(bnb.info())

# Histograms for neighborhood and room type
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(bnb['neighbourhood_group'], color="skyblue", ax=axes[0])
sns.histplot(bnb['room_type'], color="olive", ax=axes[1])
st.pyplot(fig)

# Outlier Detection
outlier_list = boxplot_stats(bnb['price']).pop(0)['fliers'].tolist()
st.write("Outlier List:", outlier_list)

# Finding the number of rows containing outliers
outlier_neighbourhood_rows = bnb[bnb['neighbourhood'].isin(outlier_list)].shape[0]
outlier_neighbourhood_group_rows = bnb[bnb['neighbourhood_group'].isin(outlier_list)].shape[0]

# Showing outlier details
st.write(f"Number of rows containing outliers in neighbourhood: {outlier_neighbourhood_rows}")
st.write(f"Percentage of outliers in neighbourhood: {(outlier_neighbourhood_rows / bnb.shape[0]) * 100:.2f}%")
st.write(f"Number of rows containing outliers in neighbourhood_group: {outlier_neighbourhood_group_rows}")
st.write(f"Percentage of outliers in neighbourhood_group: {(outlier_neighbourhood_group_rows / bnb.shape[0]) * 100:.2f}%")

# Plotting neighborhood distributions using Plotly
st.subheader("Neighborhood Group Distribution")
px.histogram(bnb, x="neighbourhood_group", nbins=60, marginal="box", title="Neighborhood Group Distribution").show()

st.subheader("Neighborhood Distribution")
px.histogram(bnb, x="neighbourhood", nbins=60, marginal="box", title="Neighborhood Distribution").show()

# Correlation Heatmap for numeric columns
st.subheader("Correlation Heatmap")
numeric_cols = bnb.select_dtypes(include=[np.number])
if not numeric_cols.empty:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_cols.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
    ax.set_title("Feature Correlation")
    st.pyplot(fig)
else:
    st.warning("No numerical columns available for correlation analysis.")

# Regression Plot for Price vs Minimum Nights
sns.regplot(x="price", y="minimum_nights", data=bnb, fit_reg=True)
plt.title("Price vs Minimum Nights")
st.pyplot()

# Linear Regression Model to Predict Prices
st.subheader("Linear Regression Model to Predict Prices")
# Selecting features for the model
X = bnb[['minimum_nights', 'neighbourhood_group', 'room_type', 'reviews_per_month', 'number_of_reviews']]
y = bnb['price']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Predicting prices
y_pred_test = lr.predict(X_test_scaled)

# Model Evaluation
st.write("Model Evaluation")
st.write(f"Coefficient of Determination (R^2): {r2_score(y_test, y_pred_test):.2f}")
st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_test):.2f}")
st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred_test):.2f}")
st.write(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f}")

# Scatterplot of Actual vs Predicted Prices
st.subheader("Scatterplot of Actual vs Predicted Prices")
fig, ax = plt.subplots(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.5, ax=ax)
plt.plot([0, max(y_test)], [0, max(y_test)], c='red', linewidth=2)
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
st.pyplot(fig)

# Boxplot of Price by Room Type
sns.boxplot(x="room_type", y="price", data=bnb)
plt.title("Price Distribution by Room Type")
st.pyplot()

# Linear Regression Model Summary
st.subheader("Model Summary")
ols_model = ols('price ~ minimum_nights + neighbourhood_group + room_type + reviews_per_month + number_of_reviews', data=bnb).fit()
st.write(ols_model.summary())

