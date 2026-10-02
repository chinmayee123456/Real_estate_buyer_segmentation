# Machine Learning Based Buyer Segmentation and Investment Profiling

## Project Overview

This project focuses on using machine learning to segment real estate buyers based on their demographic characteristics, purchasing behavior, investment purpose, financing behavior, and transaction history.

The project uses K-Means clustering as the primary segmentation technique and hierarchical clustering for comparison and validation. The resulting buyer segments are presented through an interactive Streamlit dashboard.

## Objectives

- Analyze real estate client and property transaction data.
- Integrate client and property-level information.
- Engineer buyer-level behavioral and transaction features.
- Apply categorical encoding and numerical scaling.
- Determine an appropriate number of buyer segments using the Elbow Method and Silhouette Score.
- Build buyer segments using K-Means clustering.
- Compare the results with hierarchical clustering.
- Develop an interactive dashboard for exploring buyer segments and geographic patterns.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- Jupyter Notebook

## Machine Learning

The project uses:

- K-Means Clustering
- Agglomerative Hierarchical Clustering
- Elbow Method
- Silhouette Score
- One-Hot Encoding
- StandardScaler

The final K-Means model uses **3 buyer segments**.

### Identified Buyer Segments

1. **High-Volume Multi-Property Buyers**
2. **Higher-Value Property Buyers**
3. **Lower-Value Property Buyers**

These segments are based on observed purchasing and behavioral characteristics in the dataset.

## Dashboard

The Streamlit dashboard provides:

- Dataset overview
- Buyer segment distribution
- Segment profiles
- Purchase value analysis
- Acquisition-purpose analysis
- Loan behavior
- Geographic buyer distribution
- Country and region analysis
- Interactive filters

## Project Structure

```text
Real_estate_buyer_segmentation/
│
├── app/
│   └── app.py
│
├── data/
│   ├── clients.csv
│   ├── properties.csv
│   ├── buyer_segmented.csv
│   └── buyer_segmented_dashboard.csv
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
└── README.md
