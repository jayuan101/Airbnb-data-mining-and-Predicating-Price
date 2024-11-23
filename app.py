import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
print(bnb.columns)
print(bnb.shape)
print(bnb.info())
print(bnb.describe())

# Handle missing data by dropping rows with any missing values
bnb = bnb.dropna()

# Filter numeric columns only for correlation calculation
numeric_cols = bnb.select_dtypes(include=['number']).columns
bnb_numeric = bnb[numeric_cols]

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(bnb_numeric.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Columns')
plt.show()
