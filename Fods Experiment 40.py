import numpy as np
from sklearn.linear_model import LinearRegression

# Training data
X = np.array([[1400, 3], [1600, 4], [1800, 3], [2000, 4], [2200, 5]])
y = np.array([200000, 250000, 280000, 300000, 330000])

# Train the linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# User input for the new house
house_area = int(input("Enter area of the house: "))
house_bedrooms = int(input("Enter no of bedrooms: "))
house_features = np.array([[house_area, house_bedrooms]])

# Predict the price of the new house
predict_price = lin_reg.predict(house_features)

print(f"The predicted price of the new house is: {predict_price[0]:,.2f}")
