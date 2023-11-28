import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
np.random.seed(42)
control_group = np.random.normal(loc=30, scale=10, size=100)
treatment_group = np.random.normal(loc=25, scale=10, size=100)
t_stat, p_value = stats.ttest_ind(control_group, treatment_group)
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between groups.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference between groups.")
plt.figure(figsize=(8, 6))
plt.hist(control_group, alpha=0.5, label='Control Group')
plt.hist(treatment_group, alpha=0.5, label='Treatment Group')
plt.axvline(control_group.mean(), color='blue', linestyle='dashed', linewidth=2, label='Control Mean')
plt.axvline(treatment_group.mean(), color='orange', linestyle='dashed', linewidth=2, label='Treatment Mean')
plt.title('Distribution of Data for Control and Treatment Groups')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
print(f'T-statistic: {t_stat}')
print(f'P-value: {p_value}')
