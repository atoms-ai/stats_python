import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulate the dataset based on user description
np.random.seed(42)  # For reproducibility

# Generate x values
x = np.linspace(0, 2500, 500)

# Generate y values with two different distributions
y = np.concatenate([
    np.random.uniform(0, 1, int(500 * 0.04)),  # y values from 0 to 1 for x from 0 to 100
    np.random.uniform(0.5, 1, int(500 * 0.96))  # y values from 0.5 to 1 for x from 100 to 2500
])

# Add some noise
y += np.random.normal(0, 0.05, 500)  # Adding some noise
y = np.clip(y, 0, 1)  # Ensuring y values are between 0 and 1

# Plotting the simulated dataset
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, edgecolors='w', s=30)
plt.title('Simulated Dataset')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.grid(True)
plt.show()


# Fit a polynomial curve
degree = 5  # Degree of the polynomial
coefficients = np.polyfit(x, y, degree)
polynomial = np.poly1d(coefficients)

# Generate y values from the fitted polynomial
y_fit = polynomial(x)

# Plot the original data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6, edgecolors='w', s=30, label='Original Data')
plt.plot(x, y_fit, color='red', label='Polynomial Fit')
plt.title('Polynomial Curve Fitting')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.grid(True)
plt.show()

# Calculating derivatives of the polynomial
first_derivative = np.poly1d(np.polyder(coefficients, 1))
second_derivative = np.poly1d(np.polyder(coefficients, 2))

# Finding inflection points: where second derivative is zero
critical_points = np.roots(second_derivative).real
# Filter out complex roots and points outside the x range
critical_points = critical_points[(critical_points >= 0) & (critical_points <= 2500) & np.isreal(critical_points)]

# Evaluate the first derivative at the critical points to confirm inflection
inflection_points = critical_points[first_derivative(critical_points) != 0]



