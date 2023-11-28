import matplotlib.pyplot as plt

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May']
sales_values = [10000, 12000, 9000, 15000, 11000]

plt.figure(figsize=(8, 5))
plt.plot(months, sales_values, marker='o', linestyle='-', color='b', label='Monthly Sales')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.legend()
plt.grid(True)
plt.show()
