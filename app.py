import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('listings.csv')
        return data
    except FileNotFoundError:
        st.error("The dataset file 'listings.csv' is missing.")
        return None

# App title
st.title("Airbnb Data Analysis and Price Prediction")

# Load dataset
bnb = load_data()

if bnb is not None:
    # Show dataset overview
    st.header("Dataset Overview")
    st.write("Shape of the dataset:", bnb.shape)
    st.write("Dataset Columns:", bnb.columns.tolist())
    st.write(bnb.describe())
    st.write("Dataset Sample:")
    st.dataframe(bnb.head())

    # Visualizations
    st.header("Exploratory Data Analysis")

    # Neighborhood Group Distribution
    st.subheader("Distribution of Neighborhood Groups")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='neighbourhood_group', data=bnb, palette='coolwarm', ax=ax)
    ax.set_title("Neighborhood Group Distribution")
    st.pyplot(fig)

    # Room Type Distribution
    st.subheader("Distribution of Room Types")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='room_type', data=bnb, palette='viridis', ax=ax)
    ax.set_title("Room Type Distribution")
    st.pyplot(fig)

    # Price Distribution
    st.subheader("Price Distribution")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(bnb['price'], kde=True, color='blue', ax=ax)
    ax.set_title("Price Distribution")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    if not bnb.select_dtypes(include=[np.number]).empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(bnb.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)
    else:
        st.warning("No numerical columns available for correlation analysis.")

    # Regression Plot for Price and Minimum Nights
    st.subheader("Price vs Minimum Nights")
    if 'price' in bnb.columns and 'minimum_nights' in bnb.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x="price", y="minimum_nights", data=bnb, ax=ax, fit_reg=True)
        ax.set_title("Regression Plot: Price vs Minimum Nights")
        st.pyplot(fig)
    else:
        st.warning("Price or Minimum Nights column missing for analysis.")

    # Handle missing values
    st.subheader("Handling Missing Values")
    st.write("Missing Values Before Cleaning:")
    st.write(bnb.isnull().sum())
    bnb = bnb.dropna()
    st.write("Missing Values After Cleaning:")
    st.write(bnb.isnull().sum())

    # Feature selection
    st.subheader("Feature Selection and Model Training")
    features = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']
    target = 'price'

    if all(col in bnb.columns for col in features + [target]):
        X = bnb[features]
        y = bnb[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        st.write("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
        st.write("R^2 Score:", r2_score(y_test, y_pred))

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
        feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature'))
    else:
        st.error("Required features or target variable not found in the dataset.")
else:
    st.error("Dataset could not be loaded. Please ensure the file exists.")
