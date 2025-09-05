import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Occupancy Dashboard", layout="wide")
st.title("🏡 Airbnb Occupancy Dashboard")

# -----------------------------
# Load CSVs (with error handling)
# -----------------------------
@st.cache_data
def load_csvs():
    try:
        bnb = pd.read_csv('listings.csv', usecols=['id','neighbourhood_group','room_type'])
        calendar = pd.read_csv('calendar.csv', usecols=['listing_id','date','available'])
        return bnb, calendar
    except Exception as e:
        st.error(f"Error loading CSVs: {e}")
        return pd.DataFrame(), pd.DataFrame()

bnb, calendar = load_csvs()
if bnb.empty or calendar.empty:
    st.stop()
st.success("✅ CSV files loaded successfully!")

# -----------------------------
# Preprocessing
# -----------------------------
calendar['date'] = pd.to_datetime(calendar['date'], errors='coerce')
calendar = calendar.dropna(subset=['date'])
calendar['year_month'] = calendar['date'].dt.to_period('M').astype(str)
calendar['occupied'] = calendar['available'].apply(lambda x: 1 if str(x).lower()=='f' else 0)

calendar = calendar.merge(
    bnb,
    left_on='listing_id',
    right_on='id',
    how='left'
)

monthly_occupancy = (
    calendar.groupby(['year_month','neighbourhood_group','room_type'])['occupied']
    .mean()
    .reset_index()
)
monthly_occupancy['occupancy_percent'] = monthly_occupancy['occupied']*100

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
neighbourhood_groups = sorted(monthly_occupancy['neighbourhood_group'].dropna().unique())
room_types = sorted(monthly_occupancy['room_type'].dropna().unique())

selected_groups = st.sidebar.multiselect("Neighbourhood Group", neighbourhood_groups, default=neighbourhood_groups)
selected_rooms = st.sidebar.multiselect("Room Type", room_types, default=room_types)

filtered_data = monthly_occupancy[
    (monthly_occupancy['neighbourhood_group'].isin(selected_groups)) &
    (monthly_occupancy['room_type'].isin(selected_rooms))
]

if filtered_data.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# -----------------------------
# Display Data & Download
# -----------------------------
st.subheader("Filtered Monthly Occupancy Data")
st.dataframe(filtered_data)

csv = filtered_data.to_csv(index=False)
st.download_button("📥 Download CSV", csv, "filtered_monthly_occupancy.csv", "text/csv")

# -----------------------------
# Line Chart: Occupancy Trends
# -----------------------------
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

# -----------------------------
# Heatmap: Monthly Occupancy
# -----------------------------
st.subheader("Monthly Occupancy Heatmap")

heatmap_data = filtered_data.groupby(['year_month','neighbourhood_group'])['occupancy_percent'].mean().unstack(fill_value=0)

fig_heatmap = go.Figure(
    data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='YlGnBu',
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
