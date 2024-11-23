import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats
import plotly.express as px
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings("ignore")

# Streamlit Configuration
st.set_page_config(page_title="Airbnb Analysis & Price Prediction", layout="wide")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('listings.csv')

bnb = load_data()

# Basic Data Overview
st.title("Airbnb Data Analysis & Price Prediction")
st.header("Dataset Overview")
st.write("Shape of the dataset:", bnb.shape)
st.write("Columns:", bnb.columns)
st.write("Dataset Info:")
st.write(bnb.info(verbose=True))

# Descriptive Statistics
st.subheader("Descriptive Statistics")
st.write(bnb.describe())

# Visualizations
st.subheader("Exploratory Data Analysis")

# Neighborhood and Room Type Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(bnb['neighbourhood_group'], color="skyblue", ax=axes[0])
axes[0].set_title('Neighborhood Group Distribution')
sns.histplot(bnb['room_type'], color="olive", ax=axes[1])
axes[1].set_title('Room Type Distribution')
st.pyplot(fig)

# Outlier Detection in Price
outliers = boxplot_stats(bnb['price']).pop(0)['fliers']
st.write(f"Number of outliers in price: {len(outliers)}")
st.write(f"Percentage of outliers: {(len(outliers) / len(bnb)) * 100:.2f}%")

# Interactive Plotly Histograms
st.plotly_chart(px.histogram(bnb, x="neighbourhood_group", title="Neighborhood Group Distribution"))
st.plotly_chart(px.histogram(bnb, x="neighbourhood", title="Neighborhood Distribution"))

# Correlation Heatmap
st.write("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(bnb.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Price vs Room Type Boxplot
st.write("Price vs Room Type")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="room_type", y="price", data=bnb, ax=ax)
st.pyplot(fig)

# Regression Plot: Price vs Minimum Nights
st.write("Price vs Minimum Nights")
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(x="price", y="minimum_nights", data=bnb, fit_reg=True, ax=ax)
st.pyplot(fig)

# Modeling: Price Prediction
st.subheader("Price Prediction")
bnb = bnb.dropna()  # Handle missing values
X = bnb[['minimum_nights', 'reviews_per_month', 'number_of_reviews']]
y = bnb['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model Evaluation
models = [
    ('LDA', LinearDiscriminantAnalysis()),
    ('NB', GaussianNB())
]

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    st.write(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Linear Regression Summary
st.write("Linear Regression Summary (OLS)")
m = ols('price ~ minimum_nights + reviews_per_month + number_of_reviews', data=bnb).fit()
st.text(m.summary())
