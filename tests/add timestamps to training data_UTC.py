import pandas as pd
import datetime
import os
import pytz

# Read the CSV file
input_file_path = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\Boptest900_2019_defaultControl\training_data.csv"
df = pd.read_csv(input_file_path)

# Convert 'time' column to datetime
df['UTC_time'] = pd.to_datetime(df['time'], unit='s')

# Create local time column (UTC+1)
local_tz = pytz.FixedOffset(60)  # UTC+1
df['local_time'] = df['UTC_time'].dt.tz_localize(pytz.UTC).dt.tz_convert(local_tz)

# Calculate seconds from start
start_time = df['time'].iloc[0]
df['seconds_from_start'] = df['time'] - start_time

# Reorder columns
new_cols = ['UTC_time', 'local_time', 'seconds_from_start'] + [col for col in df.columns if col not in ['UTC_time', 'local_time', 'seconds_from_start']]
df = df[new_cols]

# Create a new file name
input_dir = os.path.dirname(input_file_path)
input_filename = os.path.basename(input_file_path)
output_filename = 'timestamped_' + input_filename
output_file_path = os.path.join(input_dir, output_filename)

# Save the updated CSV to the new file
df.to_csv(output_file_path, index=False)

print(f"Updated CSV file saved as: {output_file_path}")