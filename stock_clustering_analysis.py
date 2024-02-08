import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('netflix_data.csv')

# Select relevant features for clustering
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and fit the K-means model
kmeans = KMeans(n_clusters=3, random_state=42)  # Assuming 3 clusters
kmeans.fit(X_scaled)

# Add cluster labels to the DataFrame
data['cluster'] = kmeans.labels_

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['Date'], data['Close'], c=data['cluster'], cmap='viridis')
plt.title('Stock Clusters')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()
