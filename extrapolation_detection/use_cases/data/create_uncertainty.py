import numpy as np
import pandas as pd

# Set the seed for reproducibility
np.random.seed(42)

# Number of data points to generate
num_data_points = 8787-4

# Generate random data from a normal distribution with mean 0 and std dev 1
data = np.random.normal(loc=0, scale=1, size=num_data_points)

# Create a pandas DataFrame with the generated data
df = pd.DataFrame({'Data': data})

# Save the DataFrame to an Excel file
file_name = 'gaussian_uncertainty.csv'
df.to_csv(file_name, index=False)

print(f"Data saved to '{file_name}'.")
