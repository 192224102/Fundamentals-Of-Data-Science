import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
data = {
    'Mileage': [50000, 60000, 30000, 70000, 20000],
    'Age': [3, 5, 2, 6, 1],
    'Brand': ['Toyota', 'Honda', 'Toyota', 'Ford', 'Honda'],
    'Engine_Type': ['Petrol', 'Diesel', 'Petrol', 'Diesel', 'Petrol'],
    'Price': [15000, 12000, 18000, 10000, 20000]
}
df = pd.DataFrame(data)
features = df[['Mileage', 'Age', 'Brand', 'Engine_Type']]
target = df['Price']
features = pd.get_dummies(features)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
def predict_car_price(model, input_features):
    input_df = pd.DataFrame([input_features])
    input_df = pd.get_dummies(input_df)
    missing_cols = set(X_train.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[X_train.columns]
    predicted_price = model.predict(input_df)[0]
    decision_path = model.decision_path(input_df)
    print("Decision Path:")
    for node in decision_path.indices:
        print(f"Node {node}: {model.tree_.feature[node]}, Threshold: {model.tree_.threshold[node]}")
    return predicted_price
new_car_features = {
    'Mileage': float(input("Enter Mileage: ")),
    'Age': float(input("Enter Age: ")),
    'Brand': input("Enter Brand: "),
    'Engine_Type': input("Enter Engine Type: ")
}
predicted_price = predict_car_price(model, new_car_features)
print(f"\nPredicted Price: ${predicted_price:.2f}")
