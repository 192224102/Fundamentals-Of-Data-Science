import pandas as pd
import numpy as np
item_prices = [2.5, 1.0, 3.75, 2.0]
quantities = [3, 2, 1, 4]
discount_rate = 10
tax_rate = 8
data = {'Item Price': item_prices, 'Quantity': quantities}
df = pd.DataFrame(data)
df['Subtotal'] = df['Item Price'] * df['Quantity']
df['Discount'] = (discount_rate / 100) * df['Subtotal']
df['Discounted Subtotal'] = df['Subtotal'] - df['Discount']
df['Tax'] = (tax_rate / 100) * df['Discounted Subtotal']
df['Total Cost'] = df['Discounted Subtotal'] + df['Tax']
print(df)
total_cost = np.sum(df['Total Cost'])
print(f"\nTotal Cost: ${total_cost:.2f}")
