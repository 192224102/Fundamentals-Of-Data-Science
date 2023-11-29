import numpy as np
from scipy import stats
conversion_rate_A = np.array([0.12, 0.15, 0.18, 0.22, 0.14, 0.19, 0.21, 0.23, 0.17, 0.20])
conversion_rate_B = np.array([0.09, 0.11, 0.14, 0.17, 0.12, 0.13, 0.16, 0.18, 0.10, 0.15])
t_statistic, p_value = stats.ttest_ind(conversion_rate_A, conversion_rate_B)
alpha = 0.05
print(f'T-Statistic: {t_statistic}')
print(f'P-Value: {p_value}')
if p_value < alpha:
    print("Reject the null hypothesis. There is a statistically significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no statistically significant difference.")
