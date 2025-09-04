import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Occupancy Dashboard", layout="wide")
st.title("Airbnb Data ETL & Visualization")

# ============================
# Load CSV Files
# ============================
try:
    bnb = pd.read_csv('listings.csv')
    calendar = pd.read_csv('calendar.csv')
    st.success("CSV files loaded successfully!")
except FileNotFoundError:
    st.error("Error: CSV files not found. Make sure 'listings.csv' and 'calendar.csv' are in the same folder as this app.")
    st.stop()

# ============================
# Check required columns
# ============================
required_listings_cols = ['id','neighbourhood_group','room_type']
required_calendar_cols = ['listing_id','date','available']

missing_listings = [c for c in required_listings_cols if c not in bnb.columns]
missing_calendar = [c for c in required_calendar_cols if c not in calendar.columns]

if missing_listings:
    st.error(f"Missing columns in listings.csv: {missing_listings}")
    st.stop()
if missing_calendar:
    st.error(f"Missing columns in calendar.csv: {missing_calendar}")
    st.stop()

# ============================
# Transform Data
# ============================
calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')
calendar = calendar.dropna(subset=['date'])  # remove invalid dates
calendar['year_month'] = calendar['date'].dt.to_period('M')

# Handle different values in 'available' column
calendar['occupied'] = calendar['available'].apply(lambda x: 1 if str(x).lower()=='f' else 0)

# Merge listings info
calendar = calendar.merge(
    bnb[['id','neighbourhood_group','room_type']],
    left_on='listing_id', right_on='id', how='left'
)

# Aggregate occupancy
monthly_occupancy = (
    calendar.groupby(['year_month','neighbourhood_group'])['occupied']
    .mean()
    .reset_index()
)
monthly_occupancy['occupancy_percent'] = monthly_occupancy['occupied']*100

# ============================
# Display Data
# ============================
st.subheader("Monthly Occupancy Data")
st.dataframe(monthly_occupancy)

csv = monthly_occupancy.to_csv(index=False)
st.download_button("Download Monthly Occupancy CSV", csv, "monthly_occupancy.csv", "text/csv")

# ============================
# Visualizations
# ============================
st.subheader("Seasonal Occupancy Trends")
fig = px.line(
    monthly_occupancy,
    x='year_month',
    y='occupancy_percent',
    color='neighbourhood_group',
    markers=True,
    title='Seasonal Airbnb Occupancy Trends by Neighbourhood Group'
)
fig.update_layout(xaxis_title='Month', yaxis_title='Occupancy (%)')
st.plotly_chart(fig, use_container_width=True)

# Heatmap
st.subheader("Monthly Occupancy Heatmap")
heatmap_data = monthly_occupancy.pivot(index='year_month', columns='neighbourhood_group', values='occupancy_percent')
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu')
plt.title("Monthly Occupancy (%) by Neighbourhood Group")
st.pyplot(plt)
