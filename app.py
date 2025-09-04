# ============================
# Import libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")
import pickle

# ============================
# Load dataset
# ============================
bnb = pd.read_csv('listings.csv')

# ============================
# Basic info
# ============================
print("Columns:", bnb.columns)
print("Shape:", bnb.shape)
print("Info:")
print(bnb.info())
print("Description:")
print(bnb.describe())

# ============================
# Matplotlib Histograms
# ============================
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.histplot(bnb['neighbourhood_group'], color="skyblue", ax=axes[0])
sns.histplot(bnb['room_type'], color="olive", ax=axes[1])
plt.show()

# ============================
# Outlier detection in price
# ============================
outlier_list = boxplot_stats(bnb.price)[0]['fliers']
outlier_rows = bnb[bnb.price.isin(outlier_list)]
print("Number of rows containing outliers in price:", outlier_rows.shape[0])
percent_outliers = (outlier_rows.shape[0]/bnb.shape[0])*100
print("Percentage of price outliers:", percent_outliers)

# ============================
# Plotly Histograms for categorical variables
# ============================
px.histogram(bnb, x="neighbourhood_group", title="Neighbourhood Group")
px.histogram(bnb, x="neighbourhood", title="Neighbourhood")

# ============================
# Boxplots for price vs category
# ============================
px.box(bnb, x="neighbourhood_group", y="price", color="neighbourhood_group", title="Price vs Neighbourhood Group")
px.box(bnb, x="room_type", y="price", color="room_type", title="Price vs Room Type")

# ============================
# Correlation Heatmap (numeric only)
# ============================
numeric_cols = bnb.select_dtypes(include='number')
plt.figure(figsize=(10,8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

# ============================
# Scatterplots
# ============================
sns.relplot(x="neighbourhood_group", y="price", hue="room_type", style="room_type", data=bnb)
sns.boxplot(x="room_type", y="price", data=bnb)
sns.regplot(x="price", y="minimum_nights", data=bnb, fit_reg=True)
plt.show()

# ============================
# Linear Regression
# ============================
# Simple regression
m1 = ols('price ~ minimum_nights', bnb).fit()
print("Simple Regression Summary:")
print(m1.summary())

# Multiple regression
m2 = ols('price ~ minimum_nights + neighbourhood_group + neighbourhood + room_type + reviews_per_month + number_of_reviews', bnb).fit()
print("Multiple Regression Summary:")
print(m2.summary())

# ============================
# Classification: Predict neighbourhood_group
# ============================
X = bnb.drop('neighbourhood_group', axis=1)
y = bnb['neighbourhood_group']

# Encode categorical variables
categorical_cols = X.select_dtypes(include='object').columns
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Spot check algorithms
models = [('LDA', LinearDiscriminantAnalysis()), ('NB', GaussianNB())]
results = []
names = []

for name, model in models:
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
