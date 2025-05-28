import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData
import os
import sys

# Check if running in Google Colab
IN_COLAB = 'google.colab' in sys.modules

# Setup output paths for saving results
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        # Create output directory in Google Drive
        OUTPUT_DIR = '/content/drive/MyDrive/magnetism'
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Results will be saved to Google Drive: {OUTPUT_DIR}")
    except Exception as e:
        print(f"Error mounting Google Drive: {e}")
        OUTPUT_DIR = '.'  # Fallback to current directory
else:
    # Local execution
    OUTPUT_DIR = 'magnetism_results'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Results will be saved to local directory: {OUTPUT_DIR}")
    
def get_output_path(filename):
    """Helper function to get full output path"""
    return os.path.join(OUTPUT_DIR, filename)

# Read the CSV file
if IN_COLAB:
    # Try to find the CSV file in current directory or in Google Drive
    possible_paths = [
        "magnetism_pt2.csv",
        "/content/magnetism_pt2.csv",
        "/content/drive/MyDrive/magnetism/magnetism_pt2.csv"
    ]
    
    file_found = False
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"Found data file at: {file_path}")
            data = pd.read_csv(file_path)
            file_found = True
            break
            
    if not file_found:
        raise FileNotFoundError(f"Could not find magnetism_pt2.csv in any of: {possible_paths}")
else:
    # Local execution
    file_path = "magnetism_pt2.csv"
    data = pd.read_csv(file_path)

print("Available columns:")
print(data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

def linear(A, x):
    """Linear function: y = A[1] * x + A[0]"""
    return A[1] * x + A[0]

def odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y):
    """ODR fit without bounds on parameters"""
    # Create model and data
    model = Model(fit_func)
    odr_data = RealData(x, y, sx=delta_x, sy=delta_y)

    # Create ODR instance
    odr = ODR(data=odr_data, model=model, beta0=initial_guesses)

    # Set to use least-squares method
    odr.set_job(fit_type=2)

    # Run the fit
    output = odr.run()

    # Get fitted parameters and their standard errors
    fit_params = output.beta
    fit_params_error = output.sd_beta

    return fit_params, fit_params_error, output.cov_beta, output

def calc_stats(x, y, fit_params, fit_func, output):
    """Calculate fit statistics"""
    residuals = y - fit_func(fit_params, x)
    degrees_of_freedom = len(x) - len(fit_params)
    chi2 = output.sum_square
    chi2_red = chi2/degrees_of_freedom
    p_val = stats.chi2.sf(chi2, degrees_of_freedom)
    return residuals, degrees_of_freedom, chi2_red, p_val

def remove_outliers_iterative(x, delta_x, y, delta_y, threshold=2.5, max_iterations=5):
    """
    Iteratively remove outliers based on standardized residuals from linear fit.
    
    Parameters:
    - threshold: Number of standard deviations to consider as outlier
    - max_iterations: Maximum number of iterations to prevent infinite loops
    
    Returns filtered data and indices of kept points
    """
    # Start with all data
    mask = np.ones(len(x), dtype=bool)
    iteration = 0
    
    while iteration < max_iterations:
        # Get current data
        x_current = x[mask]
        delta_x_current = delta_x[mask]
        y_current = y[mask]
        delta_y_current = delta_y[mask]
        
        if len(x_current) < 4:  # Need at least 4 points for meaningful fit
            break
            
        # Perform preliminary fit
        try:
            initial_guess = [np.mean(y_current), (y_current[-1] - y_current[0])/(x_current[-1] - x_current[0])]
            fit_params, _, _, output = odr_fit(linear, initial_guess, 
                                             x_current, delta_x_current, 
                                             y_current, delta_y_current)
            
            # Calculate residuals and standardized residuals
            residuals = y_current - linear(fit_params, x_current)
            std_residuals = np.abs(residuals) / np.std(residuals)
            
            # Find outliers
            outlier_indices = np.where(std_residuals > threshold)[0]
            
            if len(outlier_indices) == 0:
                # No more outliers found
                break
            
            # Remove the worst outlier
            worst_outlier_idx = outlier_indices[np.argmax(std_residuals[outlier_indices])]
            
            # Convert back to original indices
            original_indices = np.where(mask)[0]
            mask[original_indices[worst_outlier_idx]] = False
            
            print(f"Iteration {iteration + 1}: Removed point with standardized residual {std_residuals[worst_outlier_idx]:.2f}")
            
        except Exception as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            break
            
        iteration += 1
    
    return x[mask], delta_x[mask], y[mask], delta_y[mask], mask

def print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    """Print fit results"""
    print(f"A[0] (intercept): {fit_params[0]:.6f} ± {fit_params_error[0]:.6f}")
    print(f"A[1] (slope): {fit_params[1]:.6f} ± {fit_params_error[1]:.6f}")
    print(f"Chi Squared Reduced = {chi2_red:.5f}")
    print(f"P-value = {p_val:.5e}")
    print(f"DOF = {degrees_of_freedom}")

def perform_fit_and_plot(x_col, x_err_col, y_col, y_err_col, title_prefix, x_label, y_label, subplot_idx, remove_outliers=True):
    """
    Perform fit and create plots for given data columns
    
    Parameters:
    - remove_outliers: boolean flag to control whether to remove outliers
    """
    print(f"\n{'='*50}")
    print(f"Processing {title_prefix}")
    print(f"{'='*50}")
    
    # Extract data
    x = data[x_col].values
    delta_x = data[x_err_col].values
    y = data[y_col].values
    delta_y = data[y_err_col].values
    
    print(f"Original data points: {len(x)}")
    
    if remove_outliers:
        # Remove outliers iteratively (being very harsh with threshold=1.5)
        x_clean, delta_x_clean, y_clean, delta_y_clean, kept_mask = remove_outliers_iterative(
            x, delta_x, y, delta_y, threshold=1.5)
        
        print(f"Data points after outlier removal: {len(x_clean)}")
        print(f"Removed {len(x) - len(x_clean)} outliers")
        file_suffix = "outliers_removed"
    else:
        # Use all data without outlier removal
        x_clean, delta_x_clean, y_clean, delta_y_clean = x, delta_x, y, delta_y
        kept_mask = np.ones(len(x), dtype=bool)
        print("Using all data points (no outlier removal)")
        file_suffix = "all_data"
    
    # Initial guess for linear fit
    initial_guess = [np.mean(y_clean), (y_clean[-1] - y_clean[0])/(x_clean[-1] - x_clean[0])]
    
    # Perform the final fit
    fit_params, fit_params_error, fit_cov, output = odr_fit(
        linear, initial_guess, x_clean, delta_x_clean, y_clean, delta_y_clean)
    
    # Calculate statistics
    residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(
        x_clean, y_clean, fit_params, linear, output)
    
    # Print results
    print("\nFitting Results:")
    print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)
    
    # Create plots
    plt.style.use('classic')
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')
    
    # Create fit curve
    x_fit = np.linspace(min(x_clean) * 0.95, max(x_clean) * 1.05, 1000)
    y_fit = linear(fit_params, x_fit)
    
        # Plot 1: Data + Fit
    # Show used data points in blue
    axs[0].errorbar(x_clean, y_clean, xerr=delta_x_clean, yerr=delta_y_clean,
                    fmt='o', color='blue', label='Measured data', ecolor='gray', markersize=6)
    axs[0].plot(x_fit, y_fit, 'r-', 
                label=r'$D_{1}x+D_{0}$ fit', 
                linewidth=2)
    
    # Dynamic titles based on what we're fitting
    if 'max' in title_prefix.lower():
        main_title = "Max EMF vs. Magnet Drop Height"
        residual_title = "Residuals of Max EMF vs. Magnet Drop Height"
        residual_ylabel = r"$\mathcal{E}_{\mathrm{max}}\;-\;\mathrm{fit}(h_{\mathrm{max}})\;(\mathrm{V})$"
        scatter_xlabel = r"$h_{\mathrm{max}}\mathrm{(m)}$"
        scatter_ylabel = r"$\mathcal{E}_{\mathrm{max}}(\mathrm{V})$"
    else:
        main_title = "Min EMF vs. Magnet Drop Height"
        residual_title = "Residuals of Min EMF vs. Magnet Drop Height"
        residual_ylabel = r"$\mathcal{E}_{\mathrm{min}}\;-\;\mathrm{fit}(h_{\mathrm{min}})\;(\mathrm{V})$"
        scatter_xlabel = r"$h_{\mathrm{min}}\mathrm{(m)}$"
        scatter_ylabel = r"$\mathcal{E}_{\mathrm{min}}(\mathrm{V})$"
    
    axs[0].set_title(main_title, fontsize=13)
    axs[0].set_xlabel(scatter_xlabel)
    axs[0].set_ylabel(scatter_ylabel)
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot 2: Residuals
    axs[1].errorbar(x_clean, residuals, yerr=delta_y_clean, 
                    fmt='o', color='blue', label='Fit residuals', ecolor='gray', markersize=6)
    axs[1].axhline(y=0, color='r', linestyle='--')
    
    axs[1].set_title(residual_title, fontsize=13)
    axs[1].set_xlabel(scatter_xlabel)
    axs[1].set_ylabel(residual_ylabel)
    axs[1].grid(True)
    axs[1].legend()
    
    # Dynamic axis limits with padding
    pad_frac = 0.05
    
    # Left panel
    x_all = np.concatenate([x_clean - delta_x_clean, x_clean + delta_x_clean])
    y_all = np.concatenate([y_clean - delta_y_clean, y_clean + delta_y_clean])
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    axs[0].set_xlim(x_min - (x_max - x_min)*pad_frac,
                    x_max + (x_max - x_min)*pad_frac)
    axs[0].set_ylim(y_min - (y_max - y_min)*pad_frac,
                    y_max + (y_max - y_min)*pad_frac)
    
    # Right panel
    resid_all = np.concatenate([residuals - delta_y_clean, residuals + delta_y_clean, [0]])
    y_min2, y_max2 = resid_all.min(), resid_all.max()
    axs[1].set_xlim(x_min - (x_max - x_min)*pad_frac,
                    x_max + (x_max - x_min)*pad_frac)
    axs[1].set_ylim(y_min2 - (y_max2 - y_min2)*pad_frac,
                    y_max2 + (y_max2 - y_min2)*pad_frac)
    
    # Disable offset notation
    for ax in axs:
        ax.yaxis.get_major_formatter().set_useOffset(False)
    
    plt.tight_layout()
    plt.show()
    
    # Save individual subplots as separate figures to avoid overlapping issues
    # Save subplot 1 (data + fit)
    fig1, ax1 = plt.subplots(1, 1, figsize=(7.5, 6))
    fig1.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    # Recreate plot 1
    ax1.errorbar(x_clean, y_clean, xerr=delta_x_clean, yerr=delta_y_clean,
                 fmt='o', color='blue', label='Measured data', ecolor='gray', markersize=6)
    ax1.plot(x_fit, y_fit, 'r-', 
             label=r'$D_{1}x+D_{0}$ fit', 
             linewidth=2)
    ax1.set_title(main_title, fontsize=13)
    ax1.set_xlabel(scatter_xlabel)
    ax1.set_ylabel(scatter_ylabel)
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlim(x_min - (x_max - x_min)*pad_frac,
                 x_max + (x_max - x_min)*pad_frac)
    ax1.set_ylim(y_min - (y_max - y_min)*pad_frac,
                 y_max + (y_max - y_min)*pad_frac)
    ax1.yaxis.get_major_formatter().set_useOffset(False)
    
    plt.tight_layout()
    svg_filename = f"{title_prefix.lower().replace(' ', '_')}_subplot_1_{file_suffix}.svg"
    output_path = get_output_path(svg_filename)
    fig1.savefig(output_path)
    plt.close(fig1)
    
    # Save subplot 2 (residuals)
    fig2, ax2 = plt.subplots(1, 1, figsize=(7.5, 6))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')
    
    # Recreate plot 2
    ax2.errorbar(x_clean, residuals, yerr=delta_y_clean, 
                 fmt='o', color='blue', label='Fit residuals', ecolor='gray', markersize=6)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title(residual_title, fontsize=13)
    ax2.set_xlabel(scatter_xlabel)
    ax2.set_ylabel(residual_ylabel)
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlim(x_min - (x_max - x_min)*pad_frac,
                 x_max + (x_max - x_min)*pad_frac)
    ax2.set_ylim(y_min2 - (y_max2 - y_min2)*pad_frac,
                 y_max2 + (y_max2 - y_min2)*pad_frac)
    ax2.yaxis.get_major_formatter().set_useOffset(False)
    
    plt.tight_layout()
    svg_filename = f"{title_prefix.lower().replace(' ', '_')}_subplot_2_{file_suffix}.svg"
    output_path = get_output_path(svg_filename)
    fig2.savefig(output_path)
    plt.close(fig2)
    
    return fit_params, fit_params_error, chi2_red, p_val

# Function to save fit results to a text file
def save_results_to_file(results_max, results_min, outlier_status):
    """Save fit results to a text file"""
    if outlier_status:
        filename = "fit_results_outliers_removed.txt"
    else:
        filename = "fit_results_all_data.txt"
    
    output_path = get_output_path(filename)
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SUMMARY OF RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"Max EMF² vs Max Height:\n")
        f.write(f"  A[1] (slope): {results_max[0][1]:.6f} ± {results_max[1][1]:.6f}\n")
        f.write(f"  A[0] (intercept): {results_max[0][0]:.6f} ± {results_max[1][0]:.6f}\n")
        f.write(f"  χ²_red: {results_max[2]:.5f}, p-value: {results_max[3]:.5e}\n")
        
        f.write(f"\nMin EMF² vs Min Height:\n")
        f.write(f"  A[1] (slope): {results_min[0][1]:.6f} ± {results_min[1][1]:.6f}\n")
        f.write(f"  A[0] (intercept): {results_min[0][0]:.6f} ± {results_min[1][0]:.6f}\n")
        f.write(f"  χ²_red: {results_min[2]:.5f}, p-value: {results_min[3]:.5e}\n")
    
    print(f"Results saved to {filename}")

# Close any existing plots
plt.close('all')

# Run fits with outliers removed (harsh threshold = 1.5)
print("\n" + "="*80)
print("RUNNING FITS WITH OUTLIER REMOVAL (THRESHOLD = 1.5)")
print("="*80)

# Fit 1: ε_max^2 vs h_max with outlier removal
results_max_filtered = perform_fit_and_plot(
    x_col='h_max (m)', 
    x_err_col='Δh_max (m)',
    y_col='ε_max^2 (V)', 
    y_err_col='Δε_max^2 (V)',
    title_prefix='Max EMF Squared vs Max Height',
    x_label=r'$h_{\mathrm{max}}$ (m)',
    y_label=r'$\varepsilon_{\mathrm{max}}^2$ (V$^2$)',
    subplot_idx=1,
    remove_outliers=True
)

# Fit 2: ε_min^2 vs h_min with outlier removal
results_min_filtered = perform_fit_and_plot(
    x_col='h_min (m)', 
    x_err_col='Δh_min (m)',
    y_col='ε_min^2 (V)', 
    y_err_col='Δε_min^2 (V)',
    title_prefix='Min EMF Squared vs Min Height',
    x_label=r'$h_{\mathrm{min}}$ (m)',
    y_label=r'$\varepsilon_{\mathrm{min}}^2$ (V$^2$)',
    subplot_idx=2,
    remove_outliers=True
)

# Print and save results for filtered data
print("\n" + "="*70)
print("SUMMARY OF RESULTS (OUTLIERS REMOVED)")
print("="*70)
print(f"Max EMF² vs Max Height:")
print(f"  A[1] (slope): {results_max_filtered[0][1]:.6f} ± {results_max_filtered[1][1]:.6f}")
print(f"  A[0] (intercept): {results_max_filtered[0][0]:.6f} ± {results_max_filtered[1][0]:.6f}")
print(f"  χ²_red: {results_max_filtered[2]:.5f}, p-value: {results_max_filtered[3]:.5e}")

print(f"\nMin EMF² vs Min Height:")
print(f"  A[1] (slope): {results_min_filtered[0][1]:.6f} ± {results_min_filtered[1][1]:.6f}")
print(f"  A[0] (intercept): {results_min_filtered[0][0]:.6f} ± {results_min_filtered[1][0]:.6f}")
print(f"  χ²_red: {results_min_filtered[2]:.5f}, p-value: {results_min_filtered[3]:.5e}")

# Save filtered results to file
save_results_to_file(results_max_filtered, results_min_filtered, True)

# Run fits with all data points (no outlier removal)
print("\n" + "="*80)
print("RUNNING FITS WITH ALL DATA POINTS (NO OUTLIER REMOVAL)")
print("="*80)

# Fit 1: ε_max^2 vs h_max with all data
results_max_all = perform_fit_and_plot(
    x_col='h_max (m)', 
    x_err_col='Δh_max (m)',
    y_col='ε_max^2 (V)', 
    y_err_col='Δε_max^2 (V)',
    title_prefix='Max EMF Squared vs Max Height',
    x_label=r'$h_{\mathrm{max}}$ (m)',
    y_label=r'$\varepsilon_{\mathrm{max}}^2$ (V$^2$)',
    subplot_idx=1,
    remove_outliers=False
)

# Fit 2: ε_min^2 vs h_min with all data
results_min_all = perform_fit_and_plot(
    x_col='h_min (m)', 
    x_err_col='Δh_min (m)',
    y_col='ε_min^2 (V)', 
    y_err_col='Δε_min^2 (V)',
    title_prefix='Min EMF Squared vs Min Height',
    x_label=r'$h_{\mathrm{min}}$ (m)',
    y_label=r'$\varepsilon_{\mathrm{min}}^2$ (V$^2$)',
    subplot_idx=2,
    remove_outliers=False
)

# Print and save results for all data
print("\n" + "="*70)
print("SUMMARY OF RESULTS (ALL DATA)")
print("="*70)
print(f"Max EMF² vs Max Height:")
print(f"  A[1] (slope): {results_max_all[0][1]:.6f} ± {results_max_all[1][1]:.6f}")
print(f"  A[0] (intercept): {results_max_all[0][0]:.6f} ± {results_max_all[1][0]:.6f}")
print(f"  χ²_red: {results_max_all[2]:.5f}, p-value: {results_max_all[3]:.5e}")

print(f"\nMin EMF² vs Min Height:")
print(f"  A[1] (slope): {results_min_all[0][1]:.6f} ± {results_min_all[1][1]:.6f}")
print(f"  A[0] (intercept): {results_min_all[0][0]:.6f} ± {results_min_all[1][0]:.6f}")
print(f"  χ²_red: {results_min_all[2]:.5f}, p-value: {results_min_all[3]:.5e}")

# Save all data results to file
save_results_to_file(results_max_all, results_min_all, False)

print("\nAll fits complete!")
if IN_COLAB:
    print(f"Results and plots saved to Google Drive: {OUTPUT_DIR}")
else:
    print(f"Results and plots saved to: {OUTPUT_DIR}")