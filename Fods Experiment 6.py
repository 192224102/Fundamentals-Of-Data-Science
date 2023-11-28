import numpy as np
time_intervals = np.array([1, 2, 3, 4, 5]) 
vertical_positions = np.array([10, 15, 18, 12, 8])  
delta_vertical = np.diff(vertical_positions)
delta_time = np.diff(time_intervals)
average_velocity = delta_vertical / delta_time
print("Average Velocity:", average_velocity)
