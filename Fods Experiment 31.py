import numpy as np
from sklearn.linear_model import LogisticRegression
X_train = np.array([
    [100, 12],
    [200, 24],
])
y_train = np.array([0, 1])
model = LogisticRegression()
model.fit(X_train, y_train)
def predict_churn(new_customer_features):
    input_features = np.array(new_customer_features).reshape(1, -1)
    churn_prediction = model.predict(input_features)
    return churn_prediction[0]
usage_minutes = float(input("Enter usage minutes: "))
contract_duration = float(input("Enter contract duration (in months): "))
new_customer_features = [usage_minutes, contract_duration]
prediction = predict_churn(new_customer_features)
if prediction == 0:
    print("The new customer is predicted not to churn.")
else:
    print("The new customer is predicted to churn.")
