from scipy import stats

def check_normality(data):
    """Checks if the data is likely to be normally distributed."""
    statistic, p_value = stats.shapiro(data)
    return statistic, p_value

# Example usage
data_to_check = [0.82, 0.88, 0.85, 0.92, 0.89]
stat, p = check_normality(data_to_check)
print(f"Shapiro-Wilk statistic: {stat}, P-value: {p}")
if p > 0.05:
    print("Data looks normally distributed (fail to reject H0)")
else:
    print("Data does not look normally distributed (reject H0)")
