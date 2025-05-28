import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.odr import Model, ODR, RealData

file_path = r"/content/magnetism.xlsx" # Replace with your file path: r"/content/<your file name>.xlsx"
sheet_name = "1A" # Replace with your sheet number (1st sheet = 0)
data = pd.read_excel(file_path, sheet_name=sheet_name)
data[:5]

def choose_columns(columns):
    x = data.iloc[:,columns[0]]
    delta_x = data.iloc[:,columns[1]]
    y = data.iloc[:,columns[2]]
    delta_y = data.iloc[:,columns[3]]
    return x, delta_x, y, delta_y

def function(A, x):
    pass # Define your function. See some examples below.

def linear(A, x):
    return A[1] * x + A[0]

def parabolic(A, x):
    return A[2] * x**2 + A[1] * x + A[0]

def optics(A, x):
    return A[1] * x / (x - A[1]) + A[0]

def exponential(A, x):
    return A[2] * np.exp(A[1] * x) + A[0]

def sinusoidal(A, x):
    return A[3] * np.sin(A[1] * x + A[2]) + A[0]

def B_r(A, x):
    return A[0]/((x+A[1])**3)+A[2]

def odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y):
    model = Model(fit_func)
    odr_data = RealData(x, y, sx=delta_x, sy=delta_y)
    odr = ODR(data=odr_data, model=model, beta0=initial_guesses)
    output = odr.run()

    fit_params = output.beta
    fit_params_error = output.sd_beta
    fit_cov = output.cov_beta
    return fit_params, fit_params_error, fit_cov, output

def calc_stats(x, y, fit_params, output):
    residuals = y - fit_func(fit_params, x)
    degrees_of_freedom = len(x) - len(fit_params)
    chi2 = output.sum_square
    chi2_red = chi2/degrees_of_freedom
    p_val = stats.chi2.sf(chi2, degrees_of_freedom)
    return residuals, degrees_of_freedom, chi2_red, p_val

def print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    for i in range(len(fit_params)):
        print(f"a[{i}]: {fit_params[i]} \u00B1 {fit_params_error[i]} ")
    print(f"Chi Squared Reduced = {chi2_red:.5f} ")
    print(f"P-value = {p_val:.5e}")
    # print(f"DOF = {degrees_of_freedom}")

columns = [-4,-3,-8,-7] # Define the columns indices to represent x, delta x, y, delta y.
x, delta_x, y, delta_y = choose_columns(columns)

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='default label', ecolor='gray') # Change the label


ax.set_title('Data - add here the full title')  # Add here the full title for the fit
ax.set_xlabel('x-axis label') # Change x-axis label
ax.set_ylabel('y-axis label') # Change y-axis label

ax.grid(True)
ax.legend()

ax.ticklabel_format(style='plain', useOffset=False, axis='y')

plt.tight_layout()
plt.show()

fit_func = B_r
initial_guesses = (-2.5, -1.5, -2)
fit_params, fit_params_error, fit_cov, output = odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y)
residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(x, y, fit_params, output)
print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)

# Close any existing figures and use classic style
plt.close('all')
plt.style.use('classic')

# Create a 1×2 subplot layout
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('white')
for ax in axs:
    ax.set_facecolor('white')

# Compute a high-resolution fit curve
x_fit = np.linspace(x.min(), x.max(), 10 * len(x))
y_fit = fit_func(fit_params, x_fit)

# --- plot data + fit on left panel ---
axs[0].errorbar(x, y, xerr=delta_x, yerr=delta_y,
                fmt='.b', label='Measured data', ecolor='gray')
axs[0].plot(x_fit, y_fit, label='Sinusoidal fit', c='r', alpha=0.7)

# Titles & labels for main panel
axs[0].set_title("Measured Magnetic Field vs. Angle with Sinusoidal Fit")
axs[0].set_xlabel(r"$\theta\; \mathrm{(rad)}$")
axs[0].set_ylabel(r"$\widetilde{B}\; \mathrm{(T)}$")
axs[0].grid(True)
axs[0].legend()

# --- plot residuals on right panel ---
axs[1].errorbar(x, residuals, xerr=delta_x, yerr=delta_y,
                fmt='.b', label='Fit residuals', ecolor='gray')
axs[1].hlines(0, x.min(), x.max(),
              colors='r', linestyles='dashed')

# Titles & labels for residuals panel
axs[1].set_title("Residuals of Measured Magnetic Field vs. Angle with Sinusoidal Fit")
axs[1].set_xlabel(r"$\theta\; \mathrm{(rad)}$")
axs[1].set_ylabel(r"$\widetilde{B}\;-\;\mathrm{fit}(\theta)\;\mathrm{(T)}$")
axs[1].grid(True)
axs[1].legend()

# --- dynamic axis limits ---
pad_frac = 0.05  # 5% padding

# Left panel (data + fit)
x_all = np.concatenate([x, x_fit])
y_all = np.concatenate([y - delta_y, y + delta_y, y_fit])
x_min, x_max = x_all.min(), x_all.max()
y_min, y_max = y_all.min(), y_all.max()
axs[0].set_xlim(x_min - (x_max - x_min)*pad_frac,
                x_max + (x_max - x_min)*pad_frac)
axs[0].set_ylim(y_min - (y_max - y_min)*pad_frac,
                y_max + (y_max - y_min)*pad_frac)

# Right panel (residuals)
resid_all = np.concatenate([residuals - delta_y, residuals + delta_y, [0]])
x_min2, x_max2 = x.min(), x.max()
y_min2, y_max2 = resid_all.min(), resid_all.max()
axs[1].set_xlim(x_min2 - (x_max2 - x_min2)*pad_frac,
                x_max2 + (x_max2 - x_min2)*pad_frac)
axs[1].set_ylim(y_min2 - (y_max2 - y_min2)*pad_frac,
                y_max2 + (y_max2 - y_min2)*pad_frac)

# Disable offset notation on y‐axes
for ax in axs:
    ax.get_yaxis().get_major_formatter().set_useOffset(False)

# Final layout adjustment and display
plt.tight_layout()
plt.show()


# -----------------------
# Now: save each subplot as its own SVG
# -----------------------
# We need to draw the canvas so we can get a renderer
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

for idx, ax in enumerate(axs, start=1):
    # get the tight bbox of the axes in display coords
    bbox = ax.get_tightbbox(renderer)
    # transform to inches for bbox_inches
    inv = fig.dpi_scale_trans.inverted()
    bbox_inches = bbox.transformed(inv)
    # save just that axes
    fig.savefig(f"subplot_{idx}.svg", bbox_inches=bbox_inches)