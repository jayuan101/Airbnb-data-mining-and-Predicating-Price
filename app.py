# ============================
# Streamlit Airbnb ETL & Visualization (Direct CSV)
# ============================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Airbnb Occupancy Dashboard", layout="wide")
st.title("Airbnb Data ETL & Visualization")

# ============================
# 1. Load CSV Files
# ============================
bnb = pd.read_csv('listings.csv')
calendar = pd.read_csv('calendar.csv')

st.success("CSV files loaded successfully!")

# ============================
# 2. Transform Data
# ============================
calendar['date'] = pd.to_datetime(calendar['date'])
calendar['year_month'] = calendar['date'].dt.to_period('M')
calendar['occupied'] = calendar['available'].apply(lambda x: 1 if x=='f' else 0)

# Merge listings info
calendar = calendar.merge(
    bnb[['id','neighbourhood_group','room_type']],
    left_on='listing_id', right_on='id', how='left'
)

# Aggregate occupancy per month per neighbourhood_group
monthly_occupancy = (
    calendar.groupby(['year_month','neighbourhood_group'])['occupied']
    .mean()
    .reset_index()
)
monthly_occupancy['occupancy_percent'] = monthly_occupancy['occupied']*100

# ============================
# 3. Show Cleaned Data
# ============================
st.subheader("Monthly Occupancy Data")
st.dataframe(monthly_occupancy)

# Option to download cleaned CSV
csv = monthly_occupancy.to_csv(index=False)
st.download_button("Download Monthly Occupancy CSV", csv, "monthly_occupancy.csv", "text/csv")

# ============================
# 4. Visualization
# ============================
st.subheader("Seasonal Occupancy Trends (Interactive)")
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
heatmap_data = monthly_occupancy.pivot(
    index='year_month', columns='neighbourhood_group', values='occupancy_percent'
)
plt.figure(figsize=(12,6))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlGnBu')
plt.title("Monthly Occupancy (%) by Neighbourhood Group")
st.pyplot(plt)
