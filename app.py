import streamlit as st
import pandas as pd
import numpy as np
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

selected_groups = st.sidebar.multiselect("Neighbourhood Group", neighbourhood_groups, default=neighbourhood_groups)
selected_rooms = st.sidebar.multiselect("Room Type", room_types, default=room_types)

filtered_data = bnb[
    (bnb['neighbourhood_group'].isin(selected_groups)) &
    (bnb['room_type'].isin(selected_rooms))
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
summary = (
    filtered_data.groupby(['neighbourhood_group','room_type'])
    .agg(
        count=('id','count'),
        avg_price=('price','mean'),
        median_price=('price','median')
    )
    .reset_index()
)
st.dataframe(summary)

# ============================
# Bar Chart
# ============================
st.subheader("🏙️ Listings Count by Neighbourhood Group")
fig = px.bar(
    summary,
    x="neighbourhood_group",
    y="count",
    color="room_type",
    barmode="group",
    title="Number of Listings by Neighbourhood and Room Type"
)
st.plotly_chart(fig, use_container_width=True)

# ============================
# Box Plot
# ============================
st.subheader("💵 Price Distribution")
fig2 = px.box(
    filtered_data,
    x="neighbourhood_group",
    y="price",
    color="room_type",
    points="all",
    title="Price Distribution by Neighbourhood and Room Type"
)
st.plotly_chart(fig2, use_container_width=True)
