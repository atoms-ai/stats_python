import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulate some data
np.random.seed(0)  # For reproducibility
x = np.linspace(0, 10, 100)
y = 1 / (1 + np.exp(-x + 5)) + (np.random.rand(100) - 0.5) * 0.1

# Create a DataFrame
data = pd.DataFrame({'X': x, 'Y': y})

# Define logistic function
def logistic(x, L ,x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

# Fit the logistic model
params, _ = curve_fit(logistic, data['X'], data['Y'], p0=[1, np.median(data['X']), 1, np.min(data['Y'])])

# Create a scatterplot
sns.scatterplot(x='X', y='Y', data=data)

# Superimpose the logistic regression line
x_vals = np.linspace(data['X'].min(), data['X'].max(), 300)
y_vals = logistic(x_vals, *params)
plt.plot(x_vals, y_vals, color='red')  # Logistic regression line

# Add equation to the chart
equation_text = f'Y = {params[0]:.2f} / (1 + e^(-{params[2]:.2f}(X - {params[1]:.2f}))) + {params[3]:.2f}'
plt.text(0.5, 0.02, equation_text, ha='center', va='bottom', transform=plt.gca().transAxes, color='green')

# Show the plot
plt.show()
