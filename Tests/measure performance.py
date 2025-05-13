#measure performance test
import time
import numpy as np
import matplotlib.pyplot as plt

def measure_performance(data_sizes, num_trials=10):
    """
    Measures the time taken to load and retrieve knowledge for different data sizes.

    Args:
        data_sizes (list): A list of data sizes (number of data items).
        num_trials (int): The number of trials to run for each data size.

    Returns:
        tuple: A tuple containing lists of loading times and retrieval times.
    """
    loading_times = []
    retrieval_times = []

    for size in data_sizes:
        total_loading_time = 0
        total_retrieval_time = 0
        for _ in range(num_trials):
            # Simulate data loading
            start_time_loading = time.time()
            data = {f"item_{i}": {"name": f"Item {i}", "value": i, "category": "Category " + str(i % 3)} for i in range(size)}  # Create dummy data
            end_time_loading = time.time()
            total_loading_time += (end_time_loading - start_time_loading)

            # Simulate retrieval (assuming KnowledgeOptimizer is initialized)
            start_time_retrieval = time.time()
            # create a dummy query.
            dummy_query = "Item 1"
            # Make sure knowledge optimizer exists.
            if 'knowledge_optimizer' in globals():
              relevant_knowledge = knowledge_optimizer.retrieve_relevant_knowledge(dummy_query)
            end_time_retrieval = time.time()
            total_retrieval_time += (end_time_retrieval - start_time_retrieval)

        loading_times.append(total_loading_time / num_trials)
        retrieval_times.append(total_retrieval_time / num_trials)
    return loading_times, retrieval_times

def plot_performance(data_sizes, loading_times, retrieval_times):
    """
    Plots the loading and retrieval times for different data sizes.

    Args:
        data_sizes (list): A list of data sizes.
        loading_times (list): A list of loading times.
        retrieval_times (list): A list of retrieval times.
    """
    plt.plot(data_sizes, loading_times, label="Loading Time")
    plt.plot(data_sizes, retrieval_times, label="Retrieval Time")
    plt.xlabel("Data Size (Number of Items)")
    plt.ylabel("Time (seconds)")
    plt.title("Performance Measurement")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Example usage:
    data_sizes = [100, 1000, 10000, 100000]  # Example data sizes
    loading_times, retrieval_times = measure_performance(data_sizes)
    print("Loading Times:", loading_times)
    print("Retrieval Times:", retrieval_times)
    plot_performance(data_sizes, loading_times, retrieval_times)

![measure_performance]()
