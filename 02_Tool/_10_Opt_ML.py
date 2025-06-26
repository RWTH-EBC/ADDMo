"""
by Jun Jiang
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import DataTuning
import ModelTuning
import SharedVariables as SV
from SharedVariables import Hyperparametergrids
from BlackBoxes import RandomForestRegressor
from Functions import ModelPostprocessing, DataAnalyses
# Settings: plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['figure.dpi'] = 300
plt.rc('font', size=9)
plt.rc('axes', titlesize=9)
plt.rc('axes', labelsize=9)
plt.rc('xtick', labelsize=9)
plt.rc('ytick', labelsize=9)
plt.rc('legend', fontsize=9)
plt.rc('figure', titlesize=9)
cm = 1/2.54  # centimeters in inches

# Global var
DIR_TRIAL_INPUT = r"D:\jji\Gitce\addmo-automated-ml-regression\03_TrialInput"


def plot_comparison_scatter(
        file_output_path: str,
        meas_train: np.ndarray,
        cals_train: np.ndarray,
        meas_test: np.ndarray = None,
        cals_test: np.ndarray = None,
):
    plt.close()
    fig, ax = plt.subplots(figsize=(9*cm, 9*cm))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    # Plot the data in scatter
    ax.scatter(
        meas_train, cals_train, facecolors='none', edgecolors='#1f77b4', s=6, lw=0.8, alpha=0.7, zorder=10,
        label='training'
    )
    if meas_test is not None:
        ax.scatter(
            meas_test, cals_test, facecolors='none', edgecolors='#d62728', s=6, lw=0.8, alpha=0.7, zorder=9,
            label='testing'
        )
    # Plot the diagonal line
    ax.plot([0, 500], [0, 500], ls='-', lw=1, color='lightgrey', zorder=1)
    ax.plot([0, 500], [0, 250], ls='--', lw=1, color='lightgrey', zorder=1)
    ax.plot([0, 500], [0, 1000], ls='--', lw=1, color='lightgrey', zorder=1)

    # Set labels and title for the scatter plot
    ax.set_xlabel('Measured airflow rate in m³/h')
    ax.set_xticks(np.arange(0, 600, 50))
    ax.set_xlim((0, 250))
    ax.set_ylabel('Calculated airflow rate in m³/h')
    ax.set_yticks(np.arange(0, 600, 50))
    ax.set_ylim((0, 250))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.grid(False)
    ax.tick_params(axis='both', which='both', direction='in', length=5, grid_color='r')
    # Adjust layout
    plt.tight_layout(pad=0.5)
    plt.savefig(file_output_path)


def analyse_model_performance(dir_base: str, dir_output: str, case_name: str, use_plot: bool = False):
    # Build instance for post-processing
    post_prcs = ModelPostprocessing.PostProcessing(dir_base=dir_base)
    # Load input file for post-processing
    input_data_file_path = os.path.join(post_prcs.dir_trial_tuned_data, "ProcessedInputData_TrialTunedData.xlsx")
    df = pd.read_excel(input_data_file_path, sheet_name='ImportData')
    # Calculate volume flow using various models
    for mo in post_prcs.DEFAULT_PREDICTOR_MODEL_FILE_NAMES:
        print(f"Post-processing model {mo} ...")
        dot_Vs_cal = []
        for index, row in df.iterrows():
            data_dict = {df.columns[i]: row[i] for i in range(1, len(row) - 1)}
            dot_V_cal = post_prcs.model_single_predict(model_name=mo, features_data=data_dict)
            dot_Vs_cal.append(dot_V_cal)
        # Add result to df
        df[f'dot_V_{mo}'] = dot_Vs_cal
    # Save df to dir
    print(f"Saving df to {dir_output} ...")
    df.to_csv(os.path.join(dir_output, f'{case_name}_df.csv'), sep=";", index=False)

    # Filter df
    filter_parse = {
        'OL': {
            'train': ('2000-01-01 00:00:00', '2000-01-01 00:05:00'),
            'test': ('2000-01-01 00:30:00', '2000-01-01 00:35:00'),
        },
        'UL': {
            'train': ('2000-01-01 00:05:00', '2000-01-01 00:10:00'),
            'test': ('2000-01-01 00:35:00', '2000-01-01 00:40:00'),
        },
        'UR': {
            'train': ('2000-01-01 00:10:00', '2000-01-01 00:15:00'),
            'test': ('2000-01-01 00:40:00', '2000-01-01 00:45:00'),
        },
        'OLULUR': {
            'train': ('2000-01-01 00:00:00', '2000-01-01 00:15:00'),
            'test': ('2000-01-01 00:30:00', '2000-01-01 00:45:00'),
        },
    }
    dfs_filtered = {}
    for rm, v in filter_parse.items():
        start_time_train = pd.to_datetime(v['train'][0])
        end_time_train = pd.to_datetime(v['train'][1])
        df_train = df[(df['Time'] > start_time_train) & (df['Time'] < end_time_train)]
        start_time_test = pd.to_datetime(v['test'][0])
        end_time_test = pd.to_datetime(v['test'][1])
        df_test = df[(df['Time'] > start_time_test) & (df['Time'] < end_time_test)]
        df_total = pd.concat([df_train, df_test])
        # Add df to dfs
        dfs_filtered[rm] = {'train': df_train, 'test': df_test, 'total': df_total}

    # Analyse score, MAE, and RMSE for each model and room
    df_res_rows = []
    for mo in post_prcs.DEFAULT_PREDICTOR_MODEL_FILE_NAMES:
        for rm in ['OL', 'UL', 'UR', 'OLULUR']:
            res = {}
            for c in ['train', 'test', 'total']:
                y_act = dfs_filtered[rm][c]['dot_V [m3/h]'].to_numpy()
                y_pred = dfs_filtered[rm][c][f'dot_V_{mo}'].to_numpy()
                res.update({
                    f'score_{c}': DataAnalyses.calculate_r_squared(y_act, y_pred),
                    f'mae_{c}': DataAnalyses.calculate_mae(y_act, y_pred),
                    f'rmse_{c}': DataAnalyses.calculate_rmse(y_act, y_pred),
                })
            # Append res to row
            df_res_rows.append({
                'model': mo, 'room': rm,
                'score_train': res['score_train'], 'score_test': res['score_test'], 'score_total': res['score_total'],
                'mae_train': res['mae_train'], 'mae_test': res['mae_test'], 'mae_total': res['mae_total'],
                'rmse_train': res['rmse_train'], 'rmse_test': res['rmse_test'], 'rmse_total': res['rmse_total'],
            })
    # Generate result df
    df_res = pd.DataFrame(df_res_rows)
    # Save df to dir
    print(f"Saving df_res to {dir_output} ...")
    df_res.to_csv(os.path.join(dir_output, f'{case_name}_df_res.csv'), sep=";", index=False)

    # Plot
    if use_plot:
        # Deactivate plot function due to old version of matplotlib
        for rm in ['OL', 'UL', 'UR', 'OLULUR']:
            for mo in post_prcs.DEFAULT_PREDICTOR_MODEL_FILE_NAMES:
                print(f"Plotting result of room '{rm}' with model '{mo}' ...")
                meas_train = dfs_filtered[rm]['train']['dot_V [m3/h]'].to_numpy()
                cals_train = dfs_filtered[rm]['train'][f'dot_V_{mo}'].to_numpy()
                meas_test = dfs_filtered[rm]['test']['dot_V [m3/h]'].to_numpy()
                cals_test = dfs_filtered[rm]['test'][f'dot_V_{mo}'].to_numpy()
                plot_comparison_scatter(
                    file_output_path=os.path.join(dir_output, f"{case_name}_{rm}_{mo}.png"),
                    meas_train=meas_train, cals_train=cals_train, meas_test=meas_test, cals_test=cals_test
                )


def run_auto_tuning(
        file_trial_input: str,
        dir_data_tuning_output: str = "TrialInput",
        column_of_signal: int = 7
):
    # Split the directory path
    features_str, case_name = os.path.split(dir_data_tuning_output)

    # Configurations for data tuning
    SV.InputData = file_trial_input
    SV.NameOfData = dir_data_tuning_output
    SV.NameOfExperiment = "TrialTunedData"
    SV.ColumnOfSignal = column_of_signal
    # Preprocessing
    SV.NaNDealing = "bfill"
    SV.Resample = False
    SV.InitManFeatureSelect = False
    SV.StandardScaling = True
    SV.RobustScaling = False
    SV.NoScaling = False
    # Period selection
    SV.TimeSeriesPlot = False
    SV.ManSelect = False
    # Feature construction
    SV.Cross_auto_cloud_correlation_plotting = False
    SV.DifferenceCreate = False
    SV.ManOwnlagCreate = False
    SV.AutomaticTimeSeriesOwnlagConstruct = False
    SV.ManFeaturelagCreate = False
    SV.AutoFeaturelagCreate = False
    SV.ManFeatureSelect = False
    SV.LowVarianceFilter = False
    SV.ICA = False
    SV.UnivariateFilter = False
    SV.EmbeddedFeatureSelectionThreshold = False
    SV.RecursiveFeatureSelection = False
    SV.WrapperRecursiveFeatureSelection = False
    # Set RF as model for embedded and wrapper methods
    rf = RandomForestRegressor(max_depth=10e10, random_state=0)
    SV.EstimatorEmbedded = rf
    SV.EstimatorWrapper = SV.WrapperModels["RF"]
    SV.WrapperParams = [Hyperparametergrids["RF"], None, None, False]
    SV.MinIncrease = 0

    # Run data tuning
    DataTuning.main()

    # Configurations for model tuning
    SV.NameOfSubTest = "TrialTunedModel"
    SV.StartTraining = "2000-01-01 00:00:00"
    SV.EndTraining = "2000-01-01 00:15:00"
    SV.StartTesting = "2000-01-01 00:30:00"
    SV.EndTesting = "2000-01-01 00:45:00"
    SV.GlobalMaxEval_HyParaTuning = 100
    SV.GlobalRecu = False
    SV.GlobalShuffle = False
    SV.GlobalCV_MT = KFold(n_splits=3)
    SV.OnlyHyPara_Models = ['ModelSelection']
    SV.GlobalIndivModel = 'No'

    # Run model tuning
    ModelTuning.main_OnlyHyParaOpti()

    # Post-processing
    analyse_model_performance(
        dir_base=os.path.join("Results", dir_data_tuning_output),
        dir_output=os.path.join("Results", "PostProcessing", features_str),
        case_name=case_name,
    )


def loop_auto_tuning(features_str: str = "Ta.Ua.Beta.Dt.H.W.Ageo", column_of_signal: int = 7):
    dir_post_prcs = os.path.join("Results", "PostProcessing", features_str)
    if not os.path.exists(dir_post_prcs):
        os.makedirs(dir_post_prcs)

    all_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    all_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for rt in all_ratios:
        for idx in all_indices:
            case_name = f"rt{rt}_id{idx}"
            run_auto_tuning(
                file_trial_input=os.path.join(DIR_TRIAL_INPUT, f"AddmoTrial_{features_str}", f"InputData_{case_name}.xlsx"),
                dir_data_tuning_output=os.path.join(features_str, case_name),
                column_of_signal=column_of_signal
            )


if __name__ == '__main__':
    loop_auto_tuning(features_str="Ta.Ua.Beta.Dt.H.W.Ageo", column_of_signal=7)
