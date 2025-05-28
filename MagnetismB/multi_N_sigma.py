import numpy as np
import pandas as pd

# Data
data = {
    "height_cm": [65.5, 69.0, 72.5, 76.0, 79.5, 83.0, 86.5],
    "integral": [-0.011396, -0.01197, -0.012581, -0.011045, -0.011689, -0.013047, -0.011998],
    "error": [0.000368, 0.00036667, 0.000377, 0.00043033, 0.00048167, 0.000449, 0.00044267]
}

df = pd.DataFrame(data)

# Weighted mean of the integral
weights = 1 / np.square(df["error"])
weighted_mean = np.sum(df["integral"] * weights) / np.sum(weights)

# Chi-squared
chi_squared = np.sum(((df["integral"] - weighted_mean) ** 2) / np.square(df["error"]))

# Reduced chi-squared
dof = len(df) - 1  # degrees of freedom
chi_squared_reduced = chi_squared / dof

# Equivalent N_sigma
n_sigma_multi = np.sqrt(chi_squared_reduced)

weighted_mean, chi_squared, chi_squared_reduced, n_sigma_multi
