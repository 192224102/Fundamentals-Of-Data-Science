import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = {
    'TotalAmountSpent': [100, 150, 200, 80, 120, 250, 180, 300],
    'NumItemsPurchased': [2, 3, 4, 1, 2, 5, 4, 6]
}
df = pd.DataFrame(data)
X = df[['TotalAmountSpent', 'NumItemsPurchased']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)
plt.scatter(X['TotalAmountSpent'], X['NumItemsPurchased'], c=kmeans.labels_, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')  # Plot cluster centers
plt.xlabel('Total Amount Spent')
plt.ylabel('Number of Items Purchased')
plt.title('K-Means Clustering of Customers')
plt.show()
