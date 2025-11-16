Customer Segmentation with K-Means & PCA
Overview:

This project applies unsupervised learning to group customers based on their purchasing behavior using K-Means Clustering and Principal Component Analysis (PCA).
The goal is to identify distinct customer segments to improve marketing strategies and decision-making.
A simple Streamlit app was built to allow users to upload datasets, select features, and visualize clusters interactively.

Key Concepts:
- K-Means Clustering: Groups similar customers into clusters based on feature similarity.
- PCA (Principal Component Analysis): Reduces high-dimensional data into 2D space for easier visualization.
- StandardScaler: Ensures all features contribute equally by standardizing the data.
- Streamlit Dashboard: Provides an intuitive web interface for interactive clustering.

Tech Stack:
- Language:	Python
- Data Manipulation:	pandas, numpy
- Machine Learning:	scikit-learn
- Visualization:	matplotlib, seaborn
- Web App:	streamlit

Features:
- Upload your own CSV dataset
- Select numeric features dynamically
- Automatic handling of missing values and non-numeric data
- Scaled and original data views
- PCA visualization (2D or 1D fallback)
- Download clustered dataset as CSV

Possible Improvements:
- Add interactive Plotly visualizations
- Integrate Elbow Method for automatic K selection
- Deploy to Streamlit Cloud or Render
- Combine with RFM analysis (Recency, Frequency, Monetary) for deeper segmentation

Author:
- Chukwuemeka Eugene Obiyo
- Data Scientist
- praise609@gmail.com
🧾 License

This project is released under the MIT License — feel free to use, modify, and share with attribution.
