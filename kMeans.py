import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Load data
data = pd.read_csv(r'C:\Users\jeyas\Desktop\lab\ml\code\iris.csv')
x = data.iloc[:, :2]

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 8))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Fit KMeans with 3 clusters
k_means = KMeans(n_clusters=3, n_init=10, random_state=0)
k_means.fit(x)
labels = k_means.labels_
centroids = k_means.cluster_centers_

colours = ['blue', 'yellow', 'green']

# Plot each cluster separately, with its label and color
for i, color in enumerate(colours):
    plt.scatter(
        x.iloc[labels == i, 0], x.iloc[labels == i, 1],
        c=color, label=f'Cluster {i+1}'
    )

# Plot centroids
plt.scatter(
    centroids[:, 0], centroids[:, 1],
    marker='x', s=200, linewidths=3, color='red', label='Centroids'
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
