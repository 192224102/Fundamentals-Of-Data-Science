import numpy as np
from scipy import stats
mean_reduction_drug = 5.0
std_deviation_drug = 2.0
sample_size_drug = 50
mean_reduction_placebo = 3.0
std_deviation_placebo = 1.5
sample_size_placebo = 50
sem_drug = std_deviation_drug / np.sqrt(sample_size_drug)
sem_placebo = std_deviation_placebo / np.sqrt(sample_size_placebo)
confidence_level = 0.95
critical_value = stats.norm.ppf((1 + confidence_level) / 2)
margin_of_error_drug = critical_value * sem_drug
margin_of_error_placebo = critical_value * sem_placebo
ci_drug = (mean_reduction_drug - margin_of_error_drug, mean_reduction_drug + margin_of_error_drug)
ci_placebo = (mean_reduction_placebo - margin_of_error_placebo, mean_reduction_placebo + margin_of_error_placebo)
print(f"95% Confidence Interval for Mean Reduction in Blood Pressure (Drug Group): {ci_drug}")
print(f"95% Confidence Interval for Mean Reduction in Blood Pressure (Placebo Group): {ci_placebo}")
