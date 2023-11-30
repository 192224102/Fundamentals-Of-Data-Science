import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = np.array([[100, 24], [200, 12], [50, 6], [300, 36], [150, 18], [80, 9]])
y = np.array([0, 1, 0, 1, 0, 1])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

min = int(input("Enter usage minutes: "))
duration = int(input("Enter contract duration: "))

features = np.array([[min, duration]])
predict = logreg.predict(features)

if predict[0] == 0:
    print("The new customer is not likely to churn.")
else:
    print("The new customer is likely to churn.")
