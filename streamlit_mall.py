import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

st.title("Customer Segmentation with K-Means & PCA")
st.markdown("Upload a CSV file, select features, and visualize customer clusters interactively!")

#file Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    mall = pd.read_csv(uploaded_file)
    st.success("File successfully uploaded!")
    st.write("### Preview of Uploaded Data")
    st.dataframe(mall.head())

    #column Selection
    numeric_cols = mall.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns found. Please upload a dataset with numeric features.")
    else:
        selected_features = st.multiselect("Select numeric features for clustering:", numeric_cols, default=numeric_cols[:3])

        if len(selected_features) > 0:
            data = mall[selected_features].copy()

            #handling Missing Values
            data = data.fillna(data.mean())

            #standardising Data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)

            #displaying Original & Scaled Data
            st.markdown("### Data Diagnostics")
            st.write("Original Data (first 5 rows):")
            st.dataframe(data.head())

            st.write("Scaled Data (for clustering):")
            scaled_mall = pd.DataFrame(X_scaled, columns=data.columns)
            st.dataframe(scaled_mall.head())

            #clustering
            k = st.slider("Select number of clusters (K)", 2, 10, 4)
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            data["Cluster"] = clusters

            st.success("Clustering completed!")

            #PCA Visualization
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
                plt.figure(figsize=(8, 2))
                sns.stripplot(data=data, x=data.columns[0], y=["Cluster"]*len(data), hue="Cluster", palette="viridis", s=10, legend=False)
                plt.title("1D Cluster Visualization")
                st.pyplot(plt)

            else:
                st.warning("Not enough features for visualization. Please select 1 or more numeric columns.")

            #cluster Insights
            st.markdown("### Cluster Insights")
            cluster_summary = data.groupby("Cluster").mean().round(2)
            st.dataframe(cluster_summary)

            st.markdown("Each cluster represents customers with similar patterns. For example, one cluster may group young, high-spending customers, while another might represent older, low-spending ones.")

            #add-On: Save & Download
            st.markdown("### Save Clustered Data")
            result = mall.copy()
            result["Cluster"] = clusters
            csv = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download Clustered Data as CSV", data=csv, file_name="clustered_customers.csv", mime="text/csv")

        else:
            st.warning("Please select at least one numeric feature for clustering.")

else:
    st.info("Upload a CSV file to get started.")

st.markdown('---')
st.markdown('built with love by iamphyton') 