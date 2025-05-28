import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys

# Define the fitting function as specified
def magnetic_field_model(r, A0, A1, A2):
    """Model function: A0/((r+A1)^3) + A2"""
    return A0/((r+A1)**3) + A2

# Define a function to calculate chi-squared for optimization
def chi_square(params, r_values, B_values, B_errors):
    """Calculate chi-squared for the given parameters"""
    A0, A1, A2 = params
    model_values = magnetic_field_model(r_values, A0, A1, A2)
    return np.sum(((B_values - model_values) / B_errors)**2)

# Define a function to manually explore parameter space
def grid_search(r_values, B_values, B_errors):
    """Perform a grid search to find good initial parameters"""
    best_chi2 = float('inf')
    best_params = None
    
    # Create parameter grids to search through
    A0_values = np.logspace(-1, 3, 10)  # 0.1 to 1000
    A1_values = np.logspace(-3, 0, 10)  # 0.001 to 1
    A2_values = np.linspace(-2, 5, 10)  # -2 to 5
    
    print("Starting grid search...")
    
    best_10_params = []
    
    # Try different combinations of parameters
    for A0 in A0_values:
        for A1 in A1_values:
            for A2 in A2_values:
                params = [A0, A1, A2]
                try:
                    chi2 = chi_square(params, r_values, B_values, B_errors)
                    
                    # Keep track of the best 10 parameter sets
                    if len(best_10_params) < 10:
                        best_10_params.append((chi2, params))
                        best_10_params.sort()
                    elif chi2 < best_10_params[-1][0]:
                        best_10_params[-1] = (chi2, params)
                        best_10_params.sort()
                    
                    if chi2 < best_chi2:
                        best_chi2 = chi2
                        best_params = params
                        print(f"New best: A0={A0:.4e}, A1={A1:.4f}, A2={A2:.4f}, χ²={chi2:.2f}")
                except:
                    continue
    
    print("\nTop 10 parameter sets from grid search:")
    for i, (chi2, params) in enumerate(best_10_params):
        print(f"{i+1}. A0={params[0]:.4e}, A1={params[1]:.4f}, A2={params[2]:.4f}, χ²={chi2:.2f}")
    
    return best_params, best_10_params

def main():
    # Set higher precision for printing
    np.set_printoptions(precision=8)
    
    # Read the data file with hardcoded path
    file_path = '/home/tom/Desktop/magnetism/magnetism - 1A(2).csv'
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return
    
    # Print column names for verification
    print("Columns in the dataset:", df.columns.tolist())
    
    # Extract the relevant columns
    r = df['r (m)'].values  # Distance in meters
    Btilde = df['Btilde (T)'].values  # Btilde in Tesla
    delta_Btilde = df['ΔBtilde (T)'].values  # Uncertainty in Btilde
    
    # Filter out any rows with missing data
    valid_idx = ~np.isnan(r) & ~np.isnan(Btilde) & ~np.isnan(delta_Btilde)
    r = r[valid_idx]
    Btilde = Btilde[valid_idx]
    delta_Btilde = delta_Btilde[valid_idx]
    
    # Sort data by r to ensure proper plotting
    sort_idx = np.argsort(r)
    r = r[sort_idx]
    Btilde = Btilde[sort_idx]
    delta_Btilde = delta_Btilde[sort_idx]
    
    # Print the extracted data for verification
    print("\nExtracted data for fitting:")
    for i in range(len(r)):
        print(f"r: {r[i]:.2f} m, Btilde: {Btilde[i]:.4f} T, Uncertainty: {delta_Btilde[i]:.4f} T")
    
    # Start with grid search to find good initial parameters
    best_params, best_10_params = grid_search(r, Btilde, delta_Btilde)
    
    print("\n----- STARTING OPTIMIZATION WITH MULTIPLE INITIAL GUESSES -----")
    
    best_result = None
    best_chi2 = float('inf')
    
    # Try optimization with top 3 parameter sets from grid search
    for i, (initial_chi2, initial_params) in enumerate(best_10_params[:3]):
        print(f"\nTrying optimization with initial parameters {i+1}: {initial_params}")
        
        try:
            # Use minimize with different methods
            methods = ['Nelder-Mead', 'BFGS', 'Powell']
            
            for method in methods:
                print(f"\nOptimizing with {method} method...")
                
                result = minimize(
                    chi_square,
                    initial_params,
                    args=(r, Btilde, delta_Btilde),
                    method=method,
                    options={'maxiter': 10000, 'disp': True}
                )
                
                if result.success:
                    final_chi2 = result.fun
                    final_params = result.x
                    
                    print(f"Method {method} converged:")
                    print(f"A[0] = {final_params[0]:.6e}")
                    print(f"A[1] = {final_params[1]:.6f}")
                    print(f"A[2] = {final_params[2]:.6e}")
                    print(f"Chi-squared: {final_chi2:.4f}")
                    
                    if final_chi2 < best_chi2:
                        best_chi2 = final_chi2
                        best_result = result
                else:
                    print(f"Method {method} failed to converge.")
        
        except Exception as e:
            print(f"Error during optimization: {e}")
    
    if best_result is None:
        print("\nAll optimization methods failed. Using best parameters from grid search.")
        popt = np.array(best_params)
        # Estimate uncertainties (this is just a rough approximation)
        perr = np.abs(popt) * 0.1  # 10% uncertainty as a fallback
    else:
        popt = best_result.x
        
        # Try to estimate parameter uncertainties by exploring around the minimum
        print("\nEstimating parameter uncertainties...")
        perr = np.zeros(3)
        for i in range(3):
            # Make a small change to parameter i and see how chi-square changes
            delta = popt[i] * 0.01  # 1% change
            if delta == 0:
                delta = 0.001  # If parameter is zero, use absolute change
            
            params_plus = popt.copy()
            params_plus[i] += delta
            chi2_plus = chi_square(params_plus, r, Btilde, delta_Btilde)
            
            params_minus = popt.copy()
            params_minus[i] -= delta
            chi2_minus = chi_square(params_minus, r, Btilde, delta_Btilde)
            
            # Second derivative approximation
            d2chi2_dp2 = (chi2_plus + chi2_minus - 2*best_chi2) / (delta**2)
            
            # Uncertainty is sqrt(2/d2chi2_dp2) if d2chi2_dp2 is positive
            if d2chi2_dp2 > 0:
                perr[i] = np.sqrt(2/d2chi2_dp2)
            else:
                perr[i] = np.abs(popt[i]) * 0.1  # Fallback
    
    # Calculate fit statistics
    residuals = Btilde - magnetic_field_model(r, *popt)
    chi_squared = np.sum((residuals / delta_Btilde)**2)
    dof = len(r) - 3  # 3 parameters
    reduced_chi_squared = chi_squared / dof
    
    print("\n----- FINAL FIT RESULTS -----")
    print(f"A[0] = {popt[0]:.6e} ± {perr[0]:.6e}")
    print(f"A[1] = {popt[1]:.6f} ± {perr[1]:.6f}")
    print(f"A[2] = {popt[2]:.6e} ± {perr[2]:.6e}")
    
    print(f"\nGoodness of fit:")
    print(f"Chi-squared: {chi_squared:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Reduced chi-squared: {reduced_chi_squared:.4f}")
    
    # Create a more refined x-array for smooth curve plotting
    r_fine = np.linspace(min(r)*0.9, max(r)*1.1, 200)
    Btilde_fit = magnetic_field_model(r_fine, *popt)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot data points with error bars
    plt.errorbar(r, Btilde, yerr=delta_Btilde, fmt='o', label='Data', capsize=3, markersize=6)
    
    # Plot the fitted curve
    plt.plot(r_fine, Btilde_fit, 'r-', label='Fit: A[0]/((x+A[1])^3)+A[2]', linewidth=2)
    
    # Add equation with fitted parameters to the plot
    equation = f'$B_r(r) = \\frac{{{popt[0]:.3e}}}{{(r+{popt[1]:.4f})^3}} + ({popt[2]:.3e})$'
    plt.annotate(equation, xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Add chi-squared information
    chi_sq_text = f'$\chi^2/DOF = {reduced_chi_squared:.3f}$'
    plt.annotate(chi_sq_text, xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12,
               bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('r (m)', fontsize=12)
    plt.ylabel('Btilde (T)', fontsize=12)
    plt.title('Magnetic Field vs. Distance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Calculate and plot residuals
    plt.figure(figsize=(10, 4))
    
    # Normalized residuals (pull plot)
    normalized_residuals = residuals / delta_Btilde
    
    plt.errorbar(r, normalized_residuals, yerr=np.ones_like(r), fmt='o', capsize=3, markersize=6)
    plt.axhline(y=0, color='r', linestyle='-', linewidth=2)
    plt.xlabel('r (m)', fontsize=12)
    plt.ylabel('Normalized Residuals\n(Data - Fit)/Uncertainty', fontsize=12)
    plt.title('Residuals of the Fit', fontsize=14)
    plt.grid(True)
    
    # Save the plots
    try:
        plt.figure(1)
        plt.savefig('/home/tom/Desktop/magnetism/magnetic_field_fit.png', dpi=300, bbox_inches='tight')
        plt.figure(2)
        plt.savefig('/home/tom/Desktop/magnetism/residuals_plot.png', dpi=300, bbox_inches='tight')
        print("\nPlots saved to /home/tom/Desktop/magnetism/")
    except Exception as e:
        print(f"Error saving plots: {e}")
    
    # Save the fit results to a text file
    try:
        with open('/home/tom/Desktop/magnetism/fit_results.txt', 'w') as f:
            f.write("Fit Results for B_r(r) = A[0]/((r+A[1])**3)+A[2]\n")
            f.write("-------------------------------------------------\n")
            f.write(f"A[0] = {popt[0]:.6e} ± {perr[0]:.6e}\n")
            f.write(f"A[1] = {popt[1]:.6f} ± {perr[1]:.6f}\n")
            f.write(f"A[2] = {popt[2]:.6e} ± {perr[2]:.6e}\n\n")
            f.write(f"Chi-squared: {chi_squared:.4f}\n")
            f.write(f"Degrees of freedom: {dof}\n")
            f.write(f"Reduced chi-squared: {reduced_chi_squared:.4f}\n")
            
        print("Fit results saved to /home/tom/Desktop/magnetism/fit_results.txt")
    except Exception as e:
        print(f"Error saving fit results: {e}")
    
    # Display plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()