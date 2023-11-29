import numpy as np
from scipy.stats import t
def generate_synthetic_data(size=100):
    np.random.seed(42)
    return np.random.rand(size) * 10
def calculate_confidence_interval(sample_mean, sample_std, sample_size, confidence_level):
    dof = sample_size - 1
    critical_value = t.ppf((1 + confidence_level) / 2, dof)
    margin_of_error = critical_value * (sample_std / np.sqrt(sample_size))
    
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    
    return lower_bound, upper_bound
def main():
    data = generate_synthetic_data()
    sample_size = int(input("Enter the sample size: "))
    confidence_level = float(input("Enter the confidence level (e.g., 0.95 for 95% confidence): "))
    precision = float(input("Enter the desired level of precision: "))
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    lower_bound, upper_bound = calculate_confidence_interval(sample_mean, sample_std, sample_size, confidence_level)
    print(f"\nPoint Estimate (Sample Mean): {sample_mean}")
    print(f"Confidence Interval: [{lower_bound}, {upper_bound}]")
    if (upper_bound - lower_bound) <= precision:
        print(f"\nThe confidence interval meets the desired level of precision ({precision}).")
    else:
        print(f"\nThe confidence interval does not meet the desired level of precision ({precision}).")

if __name__ == "__main__":
    main()
