import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier

X = np.random.rand(100, 5) 
y = np.random.choice([0, 1], size=100) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k = int(input("Enter the value of k (number of neighbors): "))
knn = KNeighborsClassifier(n_neighbors=k)
features = np.random.rand(1, 5) 

# Fit the model
knn.fit(X_train, y_train)

# Make predictions
predict = knn.predict(features)

# Print the result
if predict[0] == 0:
    print("The patient does not have the medical condition.")
else:
    print("The patient has the medical condition.")
