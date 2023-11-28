import pandas as pd
import numpy as np
from datetime import datetime, timedelta
np.random.seed(42)
num_rows = 100
products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
date_today = datetime.now()
date_list = [date_today - timedelta(days=x) for x in range(num_rows)]
quantity_sold = np.random.randint(1, 50, size=num_rows)
sales_data = pd.DataFrame({
    'Date': date_list,
    'Product': np.random.choice(products, size=num_rows),
    'Quantity Sold': quantity_sold
})
start_date = pd.Timestamp.now() - pd.DateOffset(months=1)
filtered_data = sales_data[sales_data['Date'] >= start_date]
product_sales = filtered_data.groupby('Product')['Quantity Sold'].sum()
sorted_products = product_sales.sort_values(ascending=False)
top_5_products = sorted_products.head(5)
print(top_5_products)
