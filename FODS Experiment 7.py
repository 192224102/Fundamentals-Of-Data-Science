import numpy as np

house_data = np.array([
    [3, 1500, 200000],
    [4, 1800, 250000],
    [5, 2000, 300000],
    [4, 1600, 220000],
    [3, 1400, 190000]
])

bedroom_condition = house_data[:, 0] > 4
houses_more_than_four_bedrooms = house_data[bedroom_condition]

average_sale_price = np.mean(houses_more_than_four_bedrooms[:, 2])

print("Average Sale Price of Houses with More than Four Bedrooms:", average_sale_price)
