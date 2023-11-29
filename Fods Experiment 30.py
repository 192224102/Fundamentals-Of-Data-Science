import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
area = float(input("Enter the area of the house: "))
bedrooms = int(input("Enter the number of bedrooms in the house: "))
data = {'area': [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700],
        'bedrooms': [3, 3, 2, 4, 2, 3, 4, 4, 3, 3],
        'price': [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000, 255000]}
dataset = pd.DataFrame(data)
X = dataset[['area', 'bedrooms']]
y = dataset['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
new_house_features = pd.DataFrame([[area, bedrooms]], columns=['area', 'bedrooms'])
predicted_price = model.predict(new_house_features)
print(f"Predicted Price for the new house: ${predicted_price[0]:,.2f}")
