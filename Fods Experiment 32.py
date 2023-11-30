import pandas as pd
from sklearn.cluster import KMeans
customer_data = {
    'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature1': [5, 3, 8, 2, 6, 1, 9, 7, 4, 10],
    'Feature2': [7, 2, 6, 1, 8, 3, 9, 5, 4, 10],
    'Feature3': [4, 8, 2, 6, 1, 9, 3, 5, 7, 10]
}
df = pd.DataFrame(customer_data)
features = df[['Feature1', 'Feature2', 'Feature3']]
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)
new_customer_features = []
for feature in features.columns:
    value = float(input(f"Enter {feature} for the new customer: "))
    new_customer_features.append(value)
new_customer_cluster = kmeans.predict([new_customer_features])[0]
print(f"\nThe new customer belongs to Cluster {new_customer_cluster + 1}")
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
print("\nCluster Members:")
for i in range(num_clusters):
    cluster_members = df[df['Cluster'] == i]['CustomerID'].tolist()
    print(f"Cluster {i + 1}: {cluster_members}")
