#Cohens_d Test: 
import numpy as np

def cohens_d(x, y):
    """Calculates Cohen's d for effect size."""
    nx = len(x)
    ny = len(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)  # Sample variance (ddof=1 for unbiased)
    var_y = np.var(y, ddof=1)
    pooled_var = ( (nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)
    return (mean_x - mean_y) / np.sqrt(pooled_var)

# Example usage
baseline = [0.7, 0.8, 0.75, 0.85, 0.78]
dynamic = [0.82, 0.88, 0.85, 0.92, 0.89]

effect_size = cohens_d(baseline, dynamic)
print(f"Cohen's d: {effect_size}")
# Interpretation:
# 0.2 = small effect, 0.5 = medium effect, 0.8 = large effect
