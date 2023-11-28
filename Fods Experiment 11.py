import numpy as np
fuel_efficiency = np.array([25, 30, 28, 35, 22])
average_fuel_efficiency = np.mean(fuel_efficiency)
print("Average Fuel Efficiency:", average_fuel_efficiency)
model1_index = 0
model2_index = 1
fuel_efficiency_model1 = fuel_efficiency[model1_index]
fuel_efficiency_model2 = fuel_efficiency[model2_index]
percentage_improvement = ((fuel_efficiency_model2 - fuel_efficiency_model1) / fuel_efficiency_model1) * 100
print(f"Percentage Improvement from Model {model1_index} to Model {model2_index}: {percentage_improvement}%")
