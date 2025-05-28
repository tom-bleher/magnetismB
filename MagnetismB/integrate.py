#!/usr/bin/env python3
import os
import glob
import numpy as np

def compute_integral(filepath):
    # Load time and voltage data, skipping header
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    # Numerical integration using trapezoidal rule
    I = np.trapz(y, x)
    # Estimate integration error via Richardson extrapolation (coarse grid)
    I_coarse = np.trapz(y[::2], x[::2])
    error = abs(I - I_coarse) / 3.0
    return I, error

def main():
    # Path to the data/A directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', 'A')

    # Find all CSV files
    csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    results = []

    for filepath in csv_files:
        base = os.path.basename(filepath)
        name, _ = os.path.splitext(base)
        # Extract numeric height (strip 'cm')
        if name.endswith('cm'):
            height_str = name[:-2]
        else:
            height_str = name
        try:
            height = float(height_str)
        except ValueError:
            # Skip files that don't follow the <number>cm.csv pattern
            continue

        integral, error = compute_integral(filepath)
        results.append((height, integral, error))

    # Write results sorted by height to integrals.csv
    out_file = os.path.join(data_dir, 'integrals.csv')
    with open(out_file, 'w') as f:
        f.write('height_cm,integral,error\n')
        for h, val, err in sorted(results):
            f.write(f'{h},{val},{err}\n')

    print(f'Results saved to {out_file}')

if __name__ == '__main__':
    main()
