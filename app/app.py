import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Real Estate Buyer Segmentation",
    page_icon="🏠",
    layout="wide"
)


# ---------------------------------------------------------
# TITLE AND INTRODUCTION
# ---------------------------------------------------------

st.title("🏠 Real Estate Buyer Segmentation")

st.write(
    "Machine Learning Based Buyer Segmentation and Investment Profiling"
)

st.info(
    "This dashboard analyzes real estate buyers using machine learning "
    "clustering to identify distinct purchasing and investment profiles."
)


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("data/buyer_segmented_dashboard.csv")


buyer_data = load_data()


# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------

st.sidebar.header("Filters")

selected_country = st.sidebar.multiselect(
    "Country",
    options=sorted(buyer_data["country"].unique()),
    default=sorted(buyer_data["country"].unique())
)

selected_region = st.sidebar.multiselect(
    "Region",
    options=sorted(buyer_data["region"].unique()),
    default=sorted(buyer_data["region"].unique())
)

selected_purpose = st.sidebar.multiselect(
    "Acquisition Purpose",
    options=sorted(buyer_data["acquisition_purpose"].unique()),
    default=sorted(buyer_data["acquisition_purpose"].unique())
)

selected_client_type = st.sidebar.multiselect(
    "Client Type",
    options=sorted(buyer_data["client_type"].unique()),
    default=sorted(buyer_data["client_type"].unique())
)


# ---------------------------------------------------------
# APPLY FILTERS
# ---------------------------------------------------------

buyer_data = buyer_data[
    buyer_data["country"].isin(selected_country)
    & buyer_data["region"].isin(selected_region)
    & buyer_data["acquisition_purpose"].isin(selected_purpose)
    & buyer_data["client_type"].isin(selected_client_type)
]


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Buyers",
    len(buyer_data)
)

col2.metric(
    "Total Segments",
    buyer_data["cluster_name"].nunique()
)

col3.metric(
    "Avg Purchase Value",
    f"${buyer_data['total_purchase_value'].mean():,.0f}"
)


# ---------------------------------------------------------
# KEY SEGMENT INSIGHTS
# ---------------------------------------------------------

st.subheader("Key Insights")

if len(buyer_data) > 0:

    largest_segment = (
        buyer_data["cluster_name"]
        .value_counts()
        .idxmax()
    )

    largest_segment_count = (
        buyer_data["cluster_name"]
        .value_counts()
        .max()
    )

    highest_property_price_segment = (
        buyer_data
        .groupby("cluster_name")["average_property_price"]
        .mean()
        .idxmax()
    )

    highest_property_price = (
        buyer_data
        .groupby("cluster_name")["average_property_price"]
        .mean()
        .max()
    )

    highest_volume_segment = (
        buyer_data
        .groupby("cluster_name")["property_count"]
        .mean()
        .idxmax()
    )

    highest_volume = (
        buyer_data
        .groupby("cluster_name")["property_count"]
        .mean()
        .max()
    )

    insight1, insight2, insight3 = st.columns(3)

    insight1.metric(
        "Largest Buyer Segment",
        largest_segment,
        f"{largest_segment_count:,} buyers"
    )

    insight2.metric(
        "Highest Avg Property Price",
        highest_property_price_segment,
        f"${highest_property_price:,.0f}"
    )

    insight3.metric(
        "Highest Avg Properties",
        highest_volume_segment,
        f"{highest_volume:.2f} properties"
    )


# ---------------------------------------------------------
# CLUSTER DISTRIBUTION AND PURCHASE VALUE
# ---------------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Buyer Segment Distribution")

    cluster_counts = (
        buyer_data["cluster_name"]
        .value_counts()
        .rename_axis("Segment")
        .reset_index(name="Buyers")
    )

    st.bar_chart(
        cluster_counts.set_index("Segment")
    )


with col2:

    st.subheader("Average Purchase Value by Segment")

    segment_value = (
        buyer_data
        .groupby("cluster_name")["total_purchase_value"]
        .mean()
        .round(2)
        .reset_index()
    )

    segment_value.columns = [
        "Segment",
        "Average Purchase Value"
    ]

    st.bar_chart(
        segment_value.set_index("Segment")
    )


# ---------------------------------------------------------
# SEGMENT PROFILE / DESCRIPTIVE STATISTICS
# ---------------------------------------------------------

st.subheader("Segment Profile")

segment_profile = (
    buyer_data
    .groupby("cluster_name")
    .agg(
        Buyers=("client_id", "count"),
        Avg_Age=("age_at_first_transaction", "mean"),
        Avg_Properties=("property_count", "mean"),
        Avg_Purchase_Value=("total_purchase_value", "mean"),
        Avg_Property_Price=("average_property_price", "mean"),
        Avg_Satisfaction=("satisfaction_score", "mean")
    )
    .round(2)
    .reset_index()
)

segment_profile = segment_profile.rename(
    columns={
        "cluster_name": "Segment",
        "Buyers": "Buyers",
        "Avg_Age": "Average Age",
        "Avg_Properties": "Average Properties",
        "Avg_Purchase_Value": "Average Purchase Value",
        "Avg_Property_Price": "Average Property Price",
        "Avg_Satisfaction": "Average Satisfaction"
    }
)

st.dataframe(
    segment_profile,
    use_container_width=True,
    hide_index=True
)


# ---------------------------------------------------------
# INVESTOR BEHAVIOR — ACQUISITION PURPOSE
# ---------------------------------------------------------

st.subheader("Acquisition Purpose by Buyer Segment")

purpose_segment = (
    buyer_data
    .groupby(
        ["cluster_name", "acquisition_purpose"]
    )
    .size()
    .unstack(fill_value=0)
)

st.bar_chart(purpose_segment)


# ---------------------------------------------------------
# INVESTOR BEHAVIOR — LOAN BEHAVIOR
# ---------------------------------------------------------

st.subheader("Loan Behavior by Buyer Segment")

loan_percentage = (
    pd.crosstab(
        buyer_data["cluster_name"],
        buyer_data["loan_applied"],
        normalize="index"
    )
    .mul(100)
    .round(1)
)

st.bar_chart(loan_percentage)


# ---------------------------------------------------------
# GEOGRAPHIC ANALYSIS
# ---------------------------------------------------------

st.subheader("Geographic Analysis")

country_counts = (
    buyer_data["country"]
    .value_counts()
    .head(10)
    .rename_axis("Country")
    .reset_index(name="Buyers")
)

st.bar_chart(
    country_counts.set_index("Country")
)


# ---------------------------------------------------------
# GEOGRAPHIC ANALYSIS — TOP REGIONS
# ---------------------------------------------------------

st.subheader("Top 10 Regions by Buyer Count")

region_counts = (
    buyer_data["region"]
    .value_counts()
    .head(10)
    .rename_axis("Region")
    .reset_index(name="Buyers")
)

st.bar_chart(
    region_counts.set_index("Region")
)

if len(region_counts) > 0:

    top_region = region_counts.iloc[0]["Region"]
    top_region_buyers = region_counts.iloc[0]["Buyers"]

    st.caption(
        f"Most represented region: {top_region} "
        f"({top_region_buyers:,} buyers)"
    )


# ---------------------------------------------------------
# GEOGRAPHIC ANALYSIS — COUNTRY MAP
# ---------------------------------------------------------

st.subheader("Geographic Buyer Distribution")

map_data = (
    buyer_data["country"]
    .replace({
        "USA": "United States",
        "UK": "United Kingdom"
    })
    .value_counts()
    .rename_axis("country")
    .reset_index(name="buyers")
)

fig = px.choropleth(
    map_data,
    locations="country",
    locationmode="country names",
    color="buyers",
    hover_name="country",
    color_continuous_scale="Blues",
    title="Buyer Distribution by Country"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ---------------------------------------------------------
# BUYER SEGMENTS BY COUNTRY
# ---------------------------------------------------------

st.subheader("Buyer Segments by Country")

top_countries = (
    buyer_data["country"]
    .value_counts()
    .head(10)
    .index
)

country_segment = (
    buyer_data[
        buyer_data["country"].isin(top_countries)
    ]
    .groupby(
        ["country", "cluster_name"]
    )
    .size()
    .unstack(fill_value=0)
)

st.bar_chart(country_segment)