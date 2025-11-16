import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Smart Clustering App", layout="wide")
st.title("Smart Clustering & Visualization App")

#upload Section
st.markdown("### Upload Your Dataset (CSV)")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    mall = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    st.write("### Preview of Uploaded Data")
    st.dataframe(mall.head())

    #data Cleaning and Feature Selection
    st.markdown("### Feature Selection and Preprocessing")

    #automatically detecting usable columns
    numeric_cols = mall.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = mall.select_dtypes(include=['object', 'category']).columns.tolist()

    #dropping identifier-like columns automatically
    id_like_cols = [col for col in mall.columns if 'id' in col.lower() or 'name' in col.lower()]

    if id_like_cols:
        st.info(f"Dropping potential ID/name columns: {id_like_cols}")
        mall = mall.drop(columns=id_like_cols, errors='ignore')
        #updating lists after dropping
        numeric_cols = mall.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = mall.select_dtypes(include=['object', 'category']).columns.tolist()

    #letting user choose features interactively
    selected_columns = st.multiselect(
        "Select features for clustering:",
        options=numeric_cols + categorical_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )

    if not selected_columns:
        st.warning("Please select at least one feature to continue.")
        st.stop()

    #subset data
    data = mall[selected_columns].copy()

    #handling missing values
    missing_ratio = data.isna().mean()
    high_missing = missing_ratio[missing_ratio > 0.3].index.tolist()  #dropping columns with >30% missing
    if high_missing:
        st.info(f"Dropping columns with >30% missing data: {high_missing}")
        data = data.drop(columns=high_missing)

    data = data.fillna(data.median(numeric_only=True))

    #encoding categorical columns safely
    data = pd.get_dummies(data, drop_first=True)

    #dropping constant or near-constant columns
    nunique = data.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        st.info(f"Removed {len(const_cols)} constant column(s): {const_cols}")
        data = data.drop(columns=const_cols)

    #ensuring there are valid features left
    if data.shape[1] == 0:
        st.error("No usable features remain after preprocessing. Try selecting different columns.")
        st.stop()

    #scaling data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    #data Diagnostics Section
    st.markdown("### Data Diagnostics")
    st.write("Original Data (first 5 rows):")
    st.dataframe(mall[selected_columns].head())

    st.write("Scaled Data (for clustering):")
    scaled_mall = pd.DataFrame(X_scaled, columns=data.columns)
    st.dataframe(scaled_mall.head())

    st.write("Numeric Columns Used:", numeric_cols)
    st.write("Categorical Columns Encoded:", categorical_cols)
    st.write("Columns Dropped:", id_like_cols + high_missing + const_cols)
    st.write(f"Final feature count: {data.shape[1]}")

    #K-Means Clustering
    st.markdown("### K-Means Clustering")
    n_clusters = st.slider("Select number of clusters (K)", 2, 10, 3)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    data["Cluster"] = clusters

    st.success(f"K-Means clustering completed with {n_clusters} clusters!")

    #clustering Size Summary
    cluster_counts = data["Cluster"].value_counts().sort_index()
    st.subheader("luster Sizes:")
    st.table(cluster_counts)

    #PCA for visualization
    st.markdown("### PCA Visualization (2D Projection)")

    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X_scaled)
        data["PCA1"], data["PCA2"] = reduced[:, 0], reduced[:, 1]

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data, x="PCA1", y="PCA2", hue="Cluster", palette="viridis", s=80)
        plt.title("Clusters Visualized in 2D Space (via PCA)")
        st.pyplot(plt)

    elif X_scaled.shape[1] == 1:
        st.warning("Only one feature selected — showing 1D cluster visualization.")
        feature_name = data.columns[0]

        plt.figure(figsize=(8, 2))
        sns.stripplot(
            x=data[feature_name],
            y=[""] * len(data),
            hue=data["Cluster"],
            palette="viridis",
            s=10,
            legend=False
        )
        plt.title(f"1D Cluster Visualization ({feature_name})")
        st.pyplot(plt)
    else:
        st.warning("PCA visualization skipped — not enough features for 2D projection.")

    #cluster Summary
    st.markdown("### Cluster Summary (Mean Values)")
    cluster_summary = data.groupby("Cluster").mean(numeric_only=True)
    st.dataframe(cluster_summary.style.background_gradient(cmap="Blues"))

else:
    st.info("Please upload a CSV file to begin.")

st.markdown('---')
st.markdown('built with love by iamphyton') 