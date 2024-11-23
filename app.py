import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit Page Configuration
st.set_page_config(page_title="Airbnb Data Analysis and Price Prediction", layout="wide")

# Load the Dataset
@st.cache_data
def load_data():
    return pd.read_csv('listings.csv')

# Load Data
try:
    bnb = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Please ensure 'listings.csv' is uploaded.")
    st.stop()

# Display Dataset Overview
st.title("Airbnb Data Analysis and Price Prediction")
st.header("Dataset Overview")
st.write("Shape of the dataset:", bnb.shape)
st.write(bnb.head())

# Dataset Information
st.subheader("Dataset Information")
st.write("Column Data Types:")
st.write(bnb.dtypes)
st.write("Missing Values Count:")
st.write(bnb.isna().sum())

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(bnb.describe())

# Visualizing Neighborhood Group and Room Type
st.subheader("Neighborhood Group and Room Type Distribution")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(bnb['neighbourhood_group'], color="skyblue", ax=axes[0])
axes[0].set_title("Neighborhood Group Distribution")
sns.histplot(bnb['room_type'], color="olive", ax=axes[1])
axes[1].set_title("Room Type Distribution")
st.pyplot(fig)

# Outlier Detection in Price Column
st.subheader("Outlier Analysis")
if 'price' in bnb.columns:
    outlier_list = boxplot_stats(bnb['price']).pop(0)['fliers'].tolist()
    st.write("Number of Outliers in Price Column:", len(outlier_list))
else:
    st.warning("Price column not found in dataset.")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_bnb = bnb.select_dtypes(include=[np.number])
if numeric_bnb.empty:
    st.warning("No numeric columns found for correlation.")
else:
    correlation_matrix = numeric_bnb.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Relationship Between Room Type and Price
st.subheader("Room Type vs Price")
if 'room_type' in bnb.columns and 'price' in bnb.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x="room_type", y="price", data=bnb, ax=ax)
    ax.set_title("Room Type vs Price")
    st.pyplot(fig)
else:
    st.warning("Room Type or Price column missing for analysis.")

# Regression Plot for Price and Minimum Nights
st.subheader("Price vs Minimum Nights")
if 'price' in bnb.columns and 'minimum_nights' in bnb.columns:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x="price", y="minimum_nights", data=bnb, ax=ax, fit_reg=True)
    ax.set_title("
