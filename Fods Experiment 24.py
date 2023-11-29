import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Generate example data (replace this with your actual data)
np.random.seed(42)  # for reproducibility
control_group = np.random.normal(loc=50, scale=10, size=100)
treatment_group = np.random.normal(loc=55, scale=10, size=100)

# Visualize the data using a histogram
plt.hist(control_group, alpha=0.5, label='Control Group')
plt.hist(treatment_group, alpha=0.5, label='Treatment Group')
plt.legend()
plt.xlabel('Measurement')
plt.ylabel('Frequency')
plt.title('Distribution of Measurements in Control and Treatment Groups')
plt.show()

# Perform a two-sample t-test to compare the means
t_stat, p_value = stats.ttest_ind(control_group, treatment_group)

# Display the results
print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')

# Determine statistical significance
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. The treatment has a statistically significant effect.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant effect.")
