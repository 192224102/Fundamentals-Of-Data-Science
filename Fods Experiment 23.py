import pandas as pd
import numpy as np
from scipy import stats

# Generate random data for demonstration
np.random.seed(42)  # Setting seed for reproducibility
ratings = np.random.randint(1, 6, 100)  # Generating 100 random ratings between 1 and 5

# Create a DataFrame
df = pd.DataFrame({'rating': ratings})

# Calculate the average rating and standard deviation
average_rating = df['rating'].mean()
std_dev = df['rating'].std()

# Calculate the confidence interval
confidence_level = 0.95
sample_size = len(df)
margin_of_error = stats.norm.ppf((1 + confidence_level) / 2) * (std_dev / np.sqrt(sample_size))

confidence_interval = (average_rating - margin_of_error, average_rating + margin_of_error)

# Display the results
print(f"Average Rating: {average_rating:.2f}")
print(f"Confidence Interval ({confidence_level * 100}%): {confidence_interval}")
