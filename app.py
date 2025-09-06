import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Airbnb Listings Dashboard", layout="wide")
st.title("🏡 Airbnb Listings Dashboard + Price & Availability Prediction")

st.write("""
Explore Airbnb listings data from your `listings.csv`  
with filters, visualizations, predicted prices, weekly availability, and estimated trip cost.
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
# Clean data
# ============================
required_cols = ['id','neighbourhood_group','room_type','price']
for col in required_cols:
    if col not in bnb.columns:
        st.error(f"Missing column in listings.csv: {col}")
        st.stop()

bnb = bnb[bnb['price'] > 0]
bnb = bnb[bnb['price'] < 1000]  # remove extreme outliers

if 'availability_365' in bnb.columns:
    bnb['days_per_week'] = bnb['availability_365'] / 52

# ============================
# Sidebar Controls
# ============================
st.sidebar.header("Filters & Inputs")

# Neighbourhood Group & Room Type
neighbourhood_groups = sorted(bnb['neighbourhood_group'].dropna().unique())
room_types = sorted(bnb['room_type'].dropna().unique())
selected_group = st.sidebar.selectbox("Neighbourhood Group", ["All"] + neighbourhood_groups)
selected_room = st.sidebar.selectbox("Room Type", ["All"] + room_types)

# Stay Duration
stay_length = st.sidebar.number_input(
    "Enter length of stay:",
    min_value=1,
    max_value=365,
    value=7,
    help="Enter the number of days, weeks, or months you plan to stay"
)

stay_unit = st.sidebar.radio(
    "Select unit:",
    ["Days", "Weeks", "Months"],
    horizontal=True
)

# ============================
# Filter Data
# ============================
filtered_data = bnb.copy()
if selected_group != "All":
    filtered_data = filtered_data[filtered_data['neighbourhood_group'] == selected_group]
if selected_room != "All":
    filtered_data = filtered_data[filtered_data['room_type'] == selected_room]

if filtered_data.empty:
    st.warning("⚠️ No data available for selected filters.")
    st.stop()

# ============================
# Sidebar: Predictions & Summary
# ============================
st.sidebar.subheader("Predictions & Summary")

# Convert stay length to days
if stay_unit == "Days":
    total_days = stay_length
elif stay_unit == "Weeks":
    total_days = stay_length * 7
else:
    total_days = stay_length * 30  # approx

# Predicted Price
predicted_price = None
if selected_group != "All" and selected_room != "All":
    group_avg = bnb.groupby(['neighbourhood_group','room_type'])['price'].mean().reset_index()
    predicted_price = group_avg[
        (group_avg['neighbourhood_group']==selected_group) &
        (group_avg['room_type']==selected_room)
    ]['price'].values[0]
    st.sidebar.info(f"Predicted nightly price: **${predicted_price:.2f}**")

# Predicted Availability
predicted_days_year = None
if 'availability_365' in bnb.columns and selected_group != "All" and selected_room != "All":
    group_avail = bnb.groupby(['neighbourhood_group','room_type'])['days_per_week'].mean().reset_index()
    predicted_days_week = group_avail[
        (group_avail['neighbourhood_group']==selected_group) &
        (group_avail['room_type']==selected_room)
    ]['days_per_week'].values[0]
    predicted_days_year = predicted_days_week * 52
    st.sidebar.info(f"Predicted availability: **{predicted_days_week:.1f} days/week** (~{predicted_days_year:.0f} days/year)")

# Estimated Trip Cost with Availability Check
if predicted_price:
    estimated_cost = predicted_price * total_days
    if predicted_days_year:
        if total_days <= predicted_days_year:
            st.sidebar.success(f"Estimated cost for {stay_length} {stay_unit}: **${estimated_cost:,.2f}** ✅ Likely available")
        else:
            st.sidebar.error(f"Estimated cost: **${estimated_cost:,.2f}** ⚠️ Stay may exceed availability")
    else:
        st.sidebar.success(f"Estimated cost for {stay_length} {stay_unit}: **${estimated_cost:,.2f}**")

# Summary Statistics
summary = {
    "Listings Count": len(filtered_data),
    "Average Price": round(filtered_data['price'].mean(), 2),
    "Median Price": round(filtered_data['price'].median(), 2)
}
if 'availability_365' in filtered_data.columns:
    summary["Avg Availability (days/year)"] = round(filtered_data['availability_365'].mean(), 2)
st.sidebar.json(summary)

# ============================
# Main Page: Data Table
# ============================
st.subheader("Filtered Listings Data")
cols_to_show = ['id','neighbourhood_group','room_type','price']
if 'availability_365' in filtered_data.columns:
    cols_to_show.append('availability_365')
st.dataframe(filtered_data[cols_to_show])

csv = filtered_data.to_csv(index=False)
st.download_button("📥 Download CSV", csv, "filtered_listings.csv", "text/csv")

# ============================
# Price Histogram
# ============================
st.subheader("💵 Price Distribution")
fig = px.histogram(
    filtered_data,
    x="price",
    nbins=50,
    title=f"Price Distribution ({selected_group} | {selected_room})"
)
st.plotly_chart(fig, use_container_width=True)

# ============================
# Interactive Map
# ============================
if 'latitude' in filtered_data.columns and 'longitude' in filtered_data.columns:
    st.subheader("🗺️ Interactive Map of Listings")

    fig_map = px.scatter_mapbox(
        filtered_data,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        hover_name="id",
        hover_data={"neighbourhood_group": True, "room_type": True, "price": True},
        zoom=10,
        height=600,
        color_continuous_scale=px.colors.cyclical.IceFire
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    st.plotly_chart(fig_map, use_container_width=True)
