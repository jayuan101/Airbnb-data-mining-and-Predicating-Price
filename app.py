import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Occupancy Dashboard", layout="wide")
st.title("Airbnb Data ETL & Interactive Visualization")

# ============================
# Load CSV Files
# ============================
try:
    bnb = pd.read_csv('listings.csv')
    calendar = pd.read_csv('calendar.csv')
    st.success("CSV files loaded successfully!")
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

# ============================
# Check required columns
# ============================
required_listings_cols = ['id','neighbourhood_group','room_type']
required_calendar_cols = ['listing_id','date','available']

for col in required_listings_cols:
    if col not in bnb.columns:
        st.error(f"Missing column in listings.csv: {col}")
        st.stop()
for col in required_calendar_cols:
    if col not in calendar.columns:
        st.error(f"Missing column in calendar.csv: {col}")
        st.stop()

# ============================
# Transform Data
# ============================
calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')
calendar = calendar.dropna(subset=['date'])
calendar['year_month'] = calendar['date'].dt.to_period('M').astype(str)  # convert to string
calendar['occupied'] = calendar['available'].apply(lambda x: 1 if str(x).lower()=='f' else 0)

calendar = calendar.merge(
    bnb[['id','neighbourhood_group','room_type']],
    left_on='listing_id', right_on='id', how='left'
)

monthly_occupancy = (
    calendar.groupby(['year_month','neighbourhood_group','room_type'])['occupied']
    .mean()
    .reset_index()
)
monthly_occupancy['occupancy_percent'] = monthly_occupancy['occupied']*100

# ============================
# Sidebar Filters
# ============================
st.sidebar.header("Filters")
neighbourhood_groups = monthly_occupancy['neighbourhood_group'].unique().tolist()
selected_groups = st.sidebar.multiselect("Neighbourhood Group", neighbourhood_groups, default=neighbourhood_groups)

room_types = monthly_occupancy['room_type'].unique().tolist()
selected_rooms = st.sidebar.multiselect("Room Type", room_types, default=room_types)

filtered_data = monthly_occupancy[
    (monthly_occupancy['neighbourhood_group'].isin(selected_groups)) &
    (monthly_occupancy['room_type'].isin(selected_rooms))
]

if filtered_data.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# ============================
# Display Data
# ============================
st.subheader("Filtered Monthly Occupancy Data")
st.dataframe(filtered_data)

csv = filtered_data.to_csv(index=False)
st.download_button("Download Filtered CSV", csv, "filtered_monthly_occupancy.csv", "text/csv")

# ============================
# Interactive Line Chart
# ============================
st.subheader("Seasonal Occupancy Trends")
fig_line = px.line(
    filtered_data,
    x='year_month',
    y='occupancy_percent',
    color='neighbourhood_group',
    line_dash='room_type',
    markers=True,
    title='Seasonal Airbnb Occupancy Trends'
)
fig_line.update_layout(xaxis_title='Month', yaxis_title='Occupancy (%)')
st.plotly_chart(fig_line, use_container_width=True)

# ============================
# Interactive Heatmap
# ============================
st.subheader("Interactive Monthly Occupancy Heatmap")

# Pivot table for heatmap
heatmap_data = filtered_data.groupby(['year_month','neighbourhood_group'])['occupancy_percent'].mean().unstack(fill_value=0)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
        hoverongaps=False,
        hovertemplate="Month: %{y}<br>Neighbourhood: %{x}<br>Occupancy: %{z:.1f}%<extra></extra>"
    )
)

fig_heatmap.update_layout(
    title="Monthly Occupancy (%) by Neighbourhood Group",
    xaxis_title="Neighbourhood Group",
    yaxis_title="Month",
    yaxis=dict(autorange='reversed')
)

st.plotly_chart(fig_heatmap, use_container_width=True)
