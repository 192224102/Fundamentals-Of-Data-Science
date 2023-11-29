import pandas as pd
data = {'customer_age': [25, 30, 28, 35, 25, 30, 35, 40, 28, 25, 30]}
sales_data = pd.DataFrame(data)
print("Original DataFrame:")
print(sales_data)
age_frequency = sales_data['customer_age'].value_counts().sort_index()
print("\nFrequency Distribution of Customer Ages:")
print(age_frequency)
