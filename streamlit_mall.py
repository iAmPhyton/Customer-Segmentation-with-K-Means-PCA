#streamlit mall app
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation using K-Means and PCA")
st.markdown("""
Upload a customer dataset (CSV).  
This app will preprocess, run PCA (when possible), and K-Means.  
It handles small/degenerate datasets and gives helpful messages.
""")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to get started.")
    st.stop()

mall = pd.read_csv(uploaded_file)
st.subheader("Preview of Uploaded Data")
st.dataframe(mall.head())

#column selection
st.markdown("### Select the features to include in clustering")
all_columns = mall.columns.tolist()
selected_columns = st.multiselect("Choose columns for clustering:", all_columns, default=None)

if not selected_columns:
    st.warning("Please select at least one column for clustering.")
    st.stop()

#a copy
data = mall[selected_columns].copy()

#dropping columns with all-NaN
data = data.dropna(axis=1, how='all')
#optionally drop rows with all NaN, but keep rows where at least one feature is present
data = data.dropna(axis=0, how='all')

#converting non-numeric columns to dummies
data = pd.get_dummies(data, drop_first=True)

#removing constant columns
nunique = data.nunique()
const_cols = nunique[nunique <= 1].index.tolist()
if const_cols:
    st.info(f"Removed {len(const_cols)} constant column(s): {const_cols}")
    data = data.drop(columns=const_cols)

n_samples, n_features = data.shape

if n_samples < 2:
    st.error("Not enough rows after preprocessing. Need at least 2 samples to cluster.")
    st.stop()
if n_features < 1:
    st.error("No usable features remain after preprocessing. Try selecting different columns.")
    st.stop()

st.write(f"Dataset after preprocessing: **{n_samples}** rows × **{n_features}** features")

#filling remaining NaNs with column median (safer for numeric)
data = data.fillna(data.median())

#standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

#choosing allowable k range based on samples
max_k = min(10, n_samples)  #can't have more clusters than samples
if max_k < 2:
    st.error("Not enough samples to form 2 clusters. Need at least 2 samples.")
    st.stop()

st.markdown("### Choose Number of Clusters")
k = st.slider("Number of clusters (K)", min_value=2, max_value=max_k, value=min(5, max_k), step=1)

#fitting KMeans
try:
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X_scaled)
except Exception as e:
    st.error(f"KMeans failed: {e}")
    st.stop()

#computing silhouette if possible (needs at least 2 clusters and samples > clusters..)
silhouette = None
try:
    if n_samples >= 2 and k >= 2 and n_samples > k:
        silhouette = silhouette_score(X_scaled, y_kmeans)
        st.write(f"**Silhouette Score:** {silhouette:.3f}")
    else:
        st.info("Silhouette score not computed because n_samples <= n_clusters.")
except Exception:
    st.info("Silhouette score could not be computed for this dataset.")

#adding cluster labels to original mall copy for download/summary
results_mall = mall.copy()
results_mall['Cluster'] = y_kmeans

#PCA for visualization: choose n_components safely
n_components = min(2, n_samples, n_features)
if n_components == 2:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    st.markdown("### Cluster Visualization (PCA Projection - 2D)")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_kmeans, palette='tab10', s=80, ax=ax)
    ax.set_title("Customer Segments (PCA + K-Means)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster", loc='best', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

elif n_components == 1:
    #1D projection: either use PCA with 1 component or pick the first feature
    pca = PCA(n_components=1)
    X_pca = pca.fit_transform(X_scaled).ravel()
    st.markdown("### Cluster Visualization (1D PCA)")
    fig, ax = plt.subplots(figsize=(8,4))
    #stripplot / jittered scatter for 1D visualization
    sns.stripplot(x=y_kmeans, y=X_pca, jitter=True, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("PCA Component 1")
    ax.set_title("1D view of clusters")
    st.pyplot(fig)
else:
    st.info("Unable to produce PCA visualization for this dataset configuration.")

#cluster summary
st.markdown("### Cluster Summary")

#separating numeric and categorical columns
numeric_cols = results_mall[selected_columns].select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in selected_columns if col not in numeric_cols]

if numeric_cols:
    st.subheader("Numeric Feature Means by Cluster")
    st.dataframe(results_mall.groupby('Cluster')[numeric_cols].mean().round(2))

if categorical_cols:
    st.subheader("Most Frequent Categories by Cluster")
    mode_summary = (
        results_mall.groupby('Cluster')[categorical_cols]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )
    st.dataframe(mode_summary)

if not numeric_cols and not categorical_cols:
    st.info("No valid columns available for summary.")

cluster_counts = results_mall['Cluster'].value_counts().sort_index()
st.write("Cluster Sizes:")
st.table(cluster_counts)

#download results
csv = results_mall.to_csv(index=False).encode('utf-8')
st.download_button("Download Clustered Data", csv, "clustered_customers.csv", "text/csv")

st.markdown('---')
st.markdown('Built with love by iamphyton')