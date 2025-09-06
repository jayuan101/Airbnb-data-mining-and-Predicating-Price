import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Listings Dashboard", layout="wide")
st.title("🏡 Airbnb Listings Dashboard")

st.write("""
This app lets you explore Airbnb listings data from your `listings.csv`.
It provides interactive filtering, summary tables, and visualizations.
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
required_listings_cols = ['id','neighbourhood_group','room_type','price']
for col in required_listings_cols:
    if col not in bnb.columns:
        st.error(f"Missing column in listings.csv: {col}")
        st.stop()

# ============================
# Sidebar Filters
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
# Display Data & Download
# ============================
st.subheader("Filtered Listings Data")
st.dataframe(filtered_data[['id','neighbourhood_group','room_type','price']])

csv = filtered_data.to_csv(index=False)
st.download_button("📥 Download CSV", csv, "filtered_listings.csv", "text/csv")

# ============================
# Summary Metrics
# ============================
st.subheader("📊 Summary Statistics")
summary = {
    "Listings Count": len(filtered_data),
    "Average Price": round(filtered_data['price'].mean(), 2),
    "Median Price": round(filtered_data['price'].median(), 2)
}
st.json(summary)

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
