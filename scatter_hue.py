import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('your_file_path.csv')  # Replace 'your_file_path.csv' with the path to your CSV file

# Create a scatter plot
plt.figure(figsize=(10, 6))
plot = sns.scatterplot(data=df, x='x', y='y', hue='z', palette='coolwarm', edgecolor='k')

# Annotate the top 10 points with the highest z values
top_points = df.nlargest(10, 'z')
for _, row in top_points.iterrows():
    plot.text(row['x'], row['y'], f"({row['z']}, {row['x']}, {row['y']})", color='black', ha='right')

plt.title('Scatter plot colored by z value')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.colorbar(label='Z Value')
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('your_file_path.csv')  # Replace 'your_file_path.csv' with the path to your CSV

# Create a scatter plot
plt.figure(figsize=(10, 6))
plot = sns.scatterplot(data=df, x='x', y='y', hue='z', palette='coolwarm', edgecolor='k')

# Group by 'x' and get the entry with the highest 'z' for each 'x'
top_points_per_x = df.loc[df.groupby('x')['z'].idxmax()]

# Annotate these points with their 'z', 'x', and 'y' values
for _, row in top_points_per_x.iterrows():
    plot.text(row['x'], row['y'], f"({row['z']}, {row['x']}, {row['y']})", color='black', ha='right')

plt.title('Scatter plot colored by z value with top z for each unique x')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.colorbar(label='Z Value')
plt.show()
