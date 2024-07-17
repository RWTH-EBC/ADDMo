import pandas as pd
import os

# Path to the CSV file
input_file = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\Boptest900_2019_defaultControl\timestamped_Boptest900_2019_defaultControl_rawdata.csv"

# Read the CSV file
df = pd.read_csv(input_file)

# Specify the columns to keep
columns_to_extract = [
    'UTC_time',
    'local_time',
    'seconds_from_start',
    'time',
    'weaSta_reaWeaTDryBul_y',
    'weaSta_reaWeaHDirNor_y'
]

# Extract the desired columns
extracted_df = df[columns_to_extract]

# Rename only the specific columns as requested
column_mapping = {
    'weaSta_reaWeaTDryBul_y': 't_amb',
    'weaSta_reaWeaHDirNor_y': 'rad_dir'
}
extracted_df = extracted_df.rename(columns=column_mapping)

# Get the directory of the original file
output_dir = os.path.dirname(input_file)

# Create the path for the new CSV file
output_file = os.path.join(output_dir, 'disturbances.csv')

# Save the extracted data to the new CSV file
extracted_df.to_csv(output_file, index=False, sep=';')

print(f"Extracted data has been saved to: {output_file}")
print(f"Columns in the new file: {extracted_df.columns.tolist()}")