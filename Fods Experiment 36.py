import numpy as np
sales_data = np.array([10000, 12000, 15000, 18000])
total_sales_year = np.sum(sales_data)
increase_percentage = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100
print(f"Total sales for the year: {total_sales_year}")
print(f"Percentage increase from Q1 to Q4: {increase_percentage:.2f}%")
