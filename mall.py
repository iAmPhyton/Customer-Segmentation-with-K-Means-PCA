import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

mall = pd.read_csv("Mall_Customers.csv")
print(mall.head())
print(mall.info())
print(mall.describe())

#encode Gender (Male=0, Female=1)
mall['Gender'] = mall['Gender'].map({'Male': 0, 'Female': 1})

#features for clustering
X = mall[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

#standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) 

#applying PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='gray', s=50)
plt.title("Data Distribution After PCA", fontweight='bold')
plt.xlabel("Principal Component 1", fontweight='bold')
plt.ylabel("Principal Component 2", fontweight='bold')
plt.show() 

#k-means clustering
#elbow Method
inertia = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#choosing optimal k (e.g., 5)
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
print("Silhouette Score:", silhouette_score(X_scaled, y_kmeans))

#visualising clusters in 2D PCA space
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_kmeans, palette='tab10', s=80)
plt.title('Customer Segments (PCA + K-Means)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()

#adding cluster labels to original dataframe
mall['Cluster'] = y_kmeans
print(mall.groupby('Cluster').mean()) 

#hierarchiacal clustering
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=5, linkage='ward')
y_hc = hc.fit_predict(X_scaled)

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_hc, palette='viridis', s=80)
plt.title("Customer Segments (Hierarchical Clustering)")
plt.show() 