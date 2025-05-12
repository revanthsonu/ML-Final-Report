from scipy import stats

def perform_statistical_test(data1, data2, test_type="ttest_ind"):
    """
    Performs a statistical test (e.g., t-test) to determine the significance of the observed differences
    between two sets of data.

    Args:
        data1 (list or numpy array): The first set of data.
        data2 (list or numpy array): The second set of data.
        test_type (str): The type of statistical test to perform.  Supported values are "ttest_ind".
            Defaults to "ttest_ind".

    Returns:
        tuple: The test statistic and p-value.  Returns None, None if the test type is invalid.
    """
    if test_type == "ttest_ind":
        t_statistic, p_value = stats.ttest_ind(data1, data2)
        return t_statistic, p_value
    else:
        print(f"Error: Invalid test type: {test_type}")
        return None, None

if __name__ == '__main__':
    # Example Usage:
    # Assume you have collected accuracy data from the baseline and dynamic loading experiments
    baseline_accuracy = [0.80, 0.85, 0.82, 0.88, 0.84]
    dynamic_accuracy_1 = [0.90, 0.92, 0.91, 0.93, 0.89]
    dynamic_accuracy_2 = [0.86, 0.88, 0.87, 0.90, 0.85]

    t_statistic_1, p_value_1 = perform_statistical_test(baseline_accuracy, dynamic_accuracy_1)
    t_statistic_2, p_value_2 = perform_statistical_test(baseline_accuracy, dynamic_accuracy_2)

    print("Comparison between Baseline and Dynamic Loading Experiment 1:")
    print(f"T-statistic: {t_statistic_1}, P-value: {p_value_1}")

    print("Comparison between Baseline and Dynamic Loading Experiment 2:")
    print(f"T-statistic: {t_statistic_2}, P-value: {p_value_2}")
