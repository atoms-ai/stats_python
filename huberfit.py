import numpy as np
from sklearn.linear_model import HuberRegressor

# Sample data
# Replace these with your actual data
x_values = np.random.uniform(0, 20000, 1000)  # example X values
y_values = np.random.uniform(0, 20000, 1000)  # example Y values

# Define the zones
zones = [(0, 250), (250, 500), (500, 750), (750, 1000), (1000, 1250), (1250, 1500), (1500, 1750), (1750, 2000)]

# Function to split data based on zones
def split_data(x, y, lower_bound, upper_bound):
    indices = (x >= lower_bound) & (x < upper_bound)
    return x[indices], y[indices]

# Function to fit Huber regressor and return the slope
def get_huber_slope(x, y):
    if len(x) > 1:  # Ensure there are enough points to fit
        x = x.reshape(-1, 1)  # Reshape for sklearn
        huber = HuberRegressor().fit(x, y)
        return huber.coef_[0]  # Slope of the regression
    else:
        return np.nan  # Return NaN if not enough points

# Analyze each zone
slopes = []
for lower_bound, upper_bound in zones:
    x_zone, y_zone = split_data(x_values, y_values, lower_bound, upper_bound)
    slope = get_huber_slope(x_zone, y_zone)
    slopes.append((f'{lower_bound}-{upper_bound}', slope))

# Print slopes for each zone
for zone, slope in slopes:
    print(f'Zone {zone}: Slope = {slope}')
