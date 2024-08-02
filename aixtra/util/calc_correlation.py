import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr


def describe_correlation(corr, method):
    abs_corr = abs(corr)
    if method == 'pearson':
        if abs_corr < 0.10:
            return "Negligible"
        elif abs_corr < 0.40:
            return "Weak"
        elif abs_corr < 0.70:
            return "Moderate"
        elif abs_corr < 0.90:
            return "Strong"
        else:
            return "Very Strong"
    elif method == 'spearman':
        if abs_corr < 0.10:
            return "Negligible"
        elif abs_corr < 0.38:
            return "Weak"
        elif abs_corr < 0.68:
            return "Moderate"
        elif abs_corr < 0.89:
            return "Strong"
        else:
            return "Very Strong"
    elif method == 'kendall':
        if abs_corr < 0.06:
            return "Negligible"
        elif abs_corr < 0.26:
            return "Weak"
        elif abs_corr < 0.49:
            return "Moderate"
        elif abs_corr < 0.71:
            return "Strong"
        else:
            return "Very Strong"

def analyze_csv(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Extract the relevant columns (2nd and 3rd columns)
    col1_name = data.columns[1]
    col2_name = data.columns[2]
    coverage_true_validity = data.iloc[:, 1]
    mpc_avg_obj_total = data.iloc[:, 2]

    # Calculate correlations and describe strength
    pearson_corr, pearson_p_value = pearsonr(coverage_true_validity, mpc_avg_obj_total)
    pearson_strength = describe_correlation(pearson_corr, 'pearson')
    pearson_r_squared = pearson_corr ** 2

    spearman_corr, spearman_p_value = spearmanr(coverage_true_validity, mpc_avg_obj_total)
    spearman_strength = describe_correlation(spearman_corr, 'spearman')

    kendall_tau_corr, kendall_tau_p_value = kendalltau(coverage_true_validity, mpc_avg_obj_total)
    kendall_tau_strength = describe_correlation(kendall_tau_corr, 'kendall')


    # Define significance level
    alpha = 0.05

    # Check significance
    pearson_significant = pearson_p_value < alpha
    spearman_significant = spearman_p_value < alpha
    kendall_tau_significant = kendall_tau_p_value < alpha

    # Print the variable names and results
    print(f"File: {file_path}")
    print(f"Variables used for correlation: {col1_name} and {col2_name}")
    print(f"Pearson correlation: {pearson_corr:.4f} ({pearson_strength}), p-value: {pearson_p_value:.4f}, significant: {pearson_significant}")
    print(f"Pearson r²: {pearson_r_squared:.4f} (explains {pearson_r_squared*100:.2f}% of the variance)")
    print(f"Spearman correlation: {spearman_corr:.4f} ({spearman_strength}), p-value: {spearman_p_value:.4f}, significant: {spearman_significant}")
    print(f"Kendall's Tau correlation: {kendall_tau_corr:.4f} ({kendall_tau_strength}), p-value: {kendall_tau_p_value:.4f}, significant: {kendall_tau_significant}")
    print('-' * 50)

    # Return the results as a dictionary
    return {
        'file': file_path,
        'variable_1': col1_name,
        'variable_2': col2_name,
        'pearson_corr': pearson_corr,
        'pearson_strength': pearson_strength,
        'pearson_p_value': pearson_p_value,
        'pearson_significant': pearson_significant,
        'pearson_r_squared': pearson_r_squared,
        'pearson_r_squared_percentage': pearson_r_squared * 100,
        'spearman_corr': spearman_corr,
        'spearman_strength': spearman_strength,
        'spearman_p_value': spearman_p_value,
        'spearman_significant': spearman_significant,
        'kendall_tau_corr': kendall_tau_corr,
        'kendall_tau_strength': kendall_tau_strength,
        'kendall_tau_p_value': kendall_tau_p_value,
        'kendall_tau_significant': kendall_tau_significant
    }

# List of CSV file paths
csv_files = [
    r"D:\00_Temp\wandb_export_2024-08-01T16_50_34.323+02_00.csv",
    r"D:\00_Temp\wandb_export_2024-08-01T16_50_20.536+02_00.csv",
    r"D:\00_Temp\wandb_export_2024-08-01T16_50_05.350+02_00.csv",
    r"D:\00_Temp\wandb_export_2024-08-01T16_39_50.152+02_00.csv"
]

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=[
    'file', 'variable_1', 'variable_2',
    'pearson_corr', 'pearson_strength', 'pearson_p_value', 'pearson_significant', 'pearson_r_squared', 'pearson_r_squared_percentage',
    'spearman_corr', 'spearman_strength', 'spearman_p_value', 'spearman_significant',
    'kendall_tau_corr', 'kendall_tau_strength', 'kendall_tau_p_value', 'kendall_tau_significant'
])

# Loop through each CSV file and analyze
for file_path in csv_files:
    result = analyze_csv(file_path)
    results_df = results_df._append(result, ignore_index=True)

# Save the results to a CSV file
results_df.to_csv(r"D:\00_Temp\correlation_results.csv", index=False)

print("Results have been saved to 'correlation_results.csv'")