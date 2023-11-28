import pandas as pd
# Assuming your data is in a CSV file named 'property_data.csv'
property_data = pd.read_csv('C:/Users/bella/OneDrive/Documents/housing1.csv')
# Now you can use the provided code to perform the analysis
average_price_by_location = property_data.groupby('location')['listing_price'].mean()
properties_more_than_four_bedrooms = property_data[property_data['number_of_bedrooms'] > 4]
num_properties_more_than_four_bedrooms = len(properties_more_than_four_bedrooms)
property_with_largest_area = property_data.loc[property_data['area_in_square_feet'].idxmax()]
# Display the results
print("1. Average Listing Price by Location:")
print(average_price_by_location)
print("\n2. Number of Properties with More Than Four Bedrooms:", num_properties_more_than_four_bedrooms)
print("\n3. Property with the Largest Area:")
print(property_with_largest_area)
