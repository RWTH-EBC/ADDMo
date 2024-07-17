import pandas as pd
import datetime
import os

# Read the CSV file
input_file_path = r"R:\_Dissertationen\mre\Diss\08_Data_Plots_Analysis\Boptest900_2019_defaultControl\training_data.csv"
df = pd.read_csv(input_file_path)

# Set the start datetime (00:15 on 01.01.2019)
start_datetime = datetime.datetime(2019, 1, 1, 0, 0, 0)

# Generate timestamps
timestamps = [start_datetime + datetime.timedelta(minutes=15*i) for i in range(len(df))]

# Add timestamp columns
df['Human_Readable_Timestamp'] = timestamps
df['Seconds_from_Start'] = [int((t - start_datetime).total_seconds()) for t in timestamps]
df['Unix_Timestamp'] = [int(t.timestamp()) for t in timestamps]

# Reorder columns to put new timestamp columns at the beginning
cols = df.columns.tolist()
new_cols = ['Human_Readable_Timestamp', 'Seconds_from_Start', 'Unix_Timestamp'] + [col for col in cols if col not in ['Human_Readable_Timestamp', 'Seconds_from_Start', 'Unix_Timestamp']]
df = df[new_cols]

# Create a new file name
input_dir = os.path.dirname(input_file_path)
input_filename = os.path.basename(input_file_path)
output_filename = 'timestamped_' + input_filename
output_file_path = os.path.join(input_dir, output_filename)

# Save the updated CSV to the new file
df.to_csv(output_file_path, index=True, sep=';', encoding='utf-8')

print(f"Updated CSV file saved as: {output_file_path}")