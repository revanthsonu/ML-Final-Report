def compare_experiments(baseline_results, dynamic_results):
    """
    Compares the results of the baseline experiment with the dynamic loading experiments.

    Args:
        baseline_results (dict): Results from the baseline experiment (e.g., accuracy, relevance).
        dynamic_results (dict): Results from the dynamic loading experiments (e.g., accuracy, relevance) for different configurations.

    Returns:
        dict: A dictionary containing the comparison results.  For example improvement in accuracy, or change in relevance.
    """
    comparison = {}
    for key in baseline_results:
        if key in dynamic_results:
            if isinstance(baseline_results[key], (int, float)) and isinstance(dynamic_results[key], (int, float)):
                comparison[key + "_improvement"] = dynamic_results[key] - baseline_results[key]
            # elif isinstance(baseline_results[key], list) and isinstance(dynamic_results[key], list):
            #     #handle comparing lists
            else:
                comparison[key + "_change"] = "N/A"  # Cannot compare
        else:
            comparison[key + "_vs_baseline"] = "Only in baseline"
    for key in dynamic_results:
        if key not in baseline_results:
            comparison[key + "_vs_baseline"] = "Only in dynamic"
    return comparison

if __name__ == '__main__':
    # Example Usage:
    baseline_results = {
        "accuracy": 0.85,
        "relevance": 0.90,
        "response_time": 0.5,
    }
    dynamic_results_1 = {
        "accuracy": 0.92,
        "relevance": 0.95,
        "response_time": 0.8,
    }
    dynamic_results_2 = {
        "accuracy": 0.90,
        "relevance": 0.93,
        "response_time": 0.7,
    }


    comparison_1 = compare_experiments(baseline_results, dynamic_results_1)
    comparison_2 = compare_experiments(baseline_results, dynamic_results_2)

    print("Comparison with Dynamic Loading Experiment 1:", comparison_1)
    print("Comparison with Dynamic Loading Experiment 2:", comparison_2)
