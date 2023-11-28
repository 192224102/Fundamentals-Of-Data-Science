import matplotlib.pyplot as plt


months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperatures = [10, 12, 15, 18, 22, 25, 28, 27, 24, 20, 15, 12]

plt.figure(figsize=(8, 6))
plt.plot(months, temperatures, marker='o', linestyle='-')

plt.title('Monthly Temperature Data')
plt.xlabel('Months')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()
