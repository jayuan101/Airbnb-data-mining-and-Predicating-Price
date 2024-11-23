import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats  
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.tree import DecisionTreeRegressor
from sklearn import ensemble
import numpy as np
import plotly.express as px 
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")
import pickle
import os

# Function to load data with error handling
def load_data():
    try:
        # Load the CSV file
        data = pd.read_csv('listings.csv')
        if data.empty:
            raise ValueError("The CSV file is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError("The file listings.csv was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the data: {e}")

# Load the data
bnb = load_data()

# Display basic information about the dataset
bnb.columns
bnb.shape
bnb.info()
bnb.describe()

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(bnb['neighbourhood_group'], color="skyblue", ax=axes[0])
sns.histplot(bnb['room_type'], color="olive", ax=axes[1])
plt.show()

# Finding the outlier values in the price column
outlier_list = boxplot_stats(bnb['price']).pop(0)['fliers'].tolist()
print(outlier_list)

# Finding the number of rows containing outliers
outlier_neighbourhood_rows = bnb[bnb.neighbourhood.isin(outlier_list)].shape[0]
print("Number of rows containing outliers in neighbourhood:", outlier_neighbourhood_rows)

# Percentage of rows which are outliers
percent_neighbourhood_outlier = (outlier_neighbourhood_rows / bnb.shape[0]) * 100
print("Percentage of outliers in neighbourhood columns:", percent_neighbourhood_outlier)

# Visualizations using Plotly
px.histogram(data_frame=bnb, x="neighbourhood_group", nbins=60, marginal="box", title="neighbourhood_group")
px.histogram(data_frame=bnb, x="neighbourhood", nbins=60, marginal="box", title="neighbourhood")

# Heatmap of correlations
heatmap = sns.heatmap(bnb.corr(), vmin=-1, vmax=1, annot=True)

# Visualizing relationships between room type, price, and neighbourhood group
sns.relplot(x="neighbourhood_group", y="price", hue="room_type", style="room_type", data=bnb)
sns.boxplot(x="room_type", y="price", data=bnb)

# Linear regression between price and minimum nights
sns.regplot(x="price", y="minimum_nights", data=bnb, fit_reg=True)
plt.show()

# Linear regression model using Statsmodels
from statsmodels.formula.api import ols
m = ols('price ~ minimum_nights', bnb).fit()
print(m.summary())

# More complex model with multiple variables
m = ols('price ~ minimum_nights + neighbourhood_group + neighbourhood + room_type + reviews_per_month + number_of_reviews', bnb).fit()
print(m.summary())

# Preparing data for machine learning model
X = bnb.drop('neighbourhood_group', axis=1)
y = bnb['neighbourhood_group']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Spot check algorithms
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean()} ({cv_results.std()})')

# Save the model for future use
filename = 'bnb_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(m, file)
