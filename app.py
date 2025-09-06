import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Listings Dashboard", layout="wide")
st.title("🏡 Airbnb Listings Dashboard + Price Prediction")

st.write("""
Explore Airbnb listings data from your `listings.csv`  
and predict **expected price** based on *Neighbourhood Group* and *Room Type*.
""")

# ============================
# Load CSV
# ============================
@st.cache_data
def load_csv(filename):
    try:
        df = pd.read_csv(filename)
        return df
    except Exception as e:
        st.error(f"Error loading {filename}: {e}")
        return pd.DataFrame()

bnb = load_csv("listings.csv")

if bnb.empty:
    st.stop()
st.success("✅ listings.csv loaded successfully!")

# ============================
# Check required columns
# ============================
required_cols = ['id','neighbourhood_group','room_type','price']
for col in required_cols:
    if col not in bnb.columns:
        st.error(f"Missing column in listings.csv: {col}")
        st.stop()

# Clean data: remove invalid prices
bnb = bnb[bnb['price'] > 0]
bnb = bnb[bnb['price'] < 1000]  # optional: remove extreme outliers

# ============================
# Sidebar Filters (dropdowns)
# ============================
st.sidebar.header("Filters")
neighbourhood_groups = sorted(bnb['neighbourhood_group'].dropna().unique())
room_types = sorted(bnb['room_type'].dropna().unique())

selected_group = st.sidebar.selectbox("Neighbourhood Group", neighbourhood_groups)
selected_room = st.sidebar.selectbox("Room Type", room_types)

filtered_data = bnb[
    (bnb['neighbourhood_group'] == selected_group) &
    (bnb['room_type'] == selected_room)
]

if filtered_data.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ============================
# Display Data
# ============================
st.subheader("Filtered Listings Data")
st.dataframe(filtered_data[['id','neighbourhood_group','room_type','price']])

# ============================
# Build Model for Prediction
# ============================
X = bnb[['neighbourhood_group','room_type']]
y = bnb['price']

# Preprocess categorical features
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['neighbourhood_group','room_type'])]
)

# Simple linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
model.fit(X, y)

# Predict price for current selection
pred_price = model.predict(pd.DataFrame([[selected_group, selected_room]],
                                        columns=['neighbourhood_group','room_type']))[0]

# ============================
# Show Prediction
# ============================
st.subheader("💡 Predicted Price")
st.success(f"Predicted average price for **{selected_group} — {selected_room}** is **${pred_price:.2f}**")

# ============================
# Histogram of Prices
# ============================
st.subheader("💵 Price Distribution")
fig = px.histogram(
    filtered_data,
    x="price",
    nbins=50,
    title=f"Price Distribution for {selected_group} — {selected_room}"
)
st.plotly_chart(fig, use_container_width=True)

# ============================
# Map (if lat/lon exist)
# ============================
if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
    st.subheader("🗺️ Map of Listings")
    st.map(filtered_data[['latitude','longitude']])
