import pandas as pd
import os

# Define the file paths
path1 = r"D:\04_GitRepos\DDMPC_GitLab\Examples\BopTest_TAir_ODE\stored_data\vital15_check_non_dyn"
path2 = r"D:\04_GitRepos\DDMPC_GitLab\Examples\BopTest_TAir_ODE\stored_data\vital15_check_non_dyn___new_save_data"


# Function to read CSV files in a directory
def read_csv_files(directory):
    data_frames = {}
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            data_frames[filename] = df
    return data_frames


# Read CSV files from both directories
dfs1 = read_csv_files(path1)
dfs2 = read_csv_files(path2)

# Compare the DataFrames
for filename in set(dfs1.keys()).union(dfs2.keys()):
    if filename in dfs1 and filename in dfs2:
        df1 = dfs1[filename]
        df2 = dfs2[filename]

        # Check if DataFrames are identical
        if df1.equals(df2):
            print(f"{filename}: The files are identical.")
        else:
            print(f"{filename}: The files are different.")

            # Check for structural differences
            if df1.shape != df2.shape:
                print(f"  - Different shapes: {df1.shape} vs {df2.shape}")
            elif list(df1.columns) != list(df2.columns):
                print("  - Different column names")
            else:
                # Compare data
                diff = (df1 != df2).sum().sum()
                print(f"  - Number of different cells: {diff}")

                # Show a sample of differences
                if diff > 0:
                    diff_df = (df1 != df2).stack()
                    diff_df = diff_df[diff_df]
                    print("  - Sample of differences:")
                    for idx, value in diff_df.head().items():
                        print(f"    Row {idx[0]}, Column '{idx[1]}':")
                        print(f"      File 1: {df1.loc[idx[0], idx[1]]}")
                        print(f"      File 2: {df2.loc[idx[0], idx[1]]}")
    else:
        print(f"{filename}: File exists in only one directory.")