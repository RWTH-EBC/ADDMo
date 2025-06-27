"""
by Jun Jiang
"""
import os
import pandas as pd
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
FILTER_PARSE = {
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


def analyse_model_performance(dir_base: str, dir_output: str, case_name: str, room_name: str):
    # Build instance for post-processing
    post_prcs = ModelPostprocessing.PostProcessing(dir_base=dir_base, dir_model=f"TrialTunedModel_{room_name}")

    # Load input file for post-processing
    input_data_file_path = os.path.join(post_prcs.dir_trial_tuned_data, "ProcessedInputData_TrialTunedData.xlsx")
    df = pd.read_excel(input_data_file_path, sheet_name='ImportData')

    # Filter df based on room_name
    start_time_train = pd.to_datetime(FILTER_PARSE[room_name]['train'][0])
    end_time_train = pd.to_datetime(FILTER_PARSE[room_name]['train'][1])
    start_time_test = pd.to_datetime(FILTER_PARSE[room_name]['test'][0])
    end_time_test = pd.to_datetime(FILTER_PARSE[room_name]['test'][1])
    df_total = df[(df['Time'] > start_time_train) & (df['Time'] < end_time_train) |
                  (df['Time'] > start_time_test) & (df['Time'] < end_time_test)].copy()

    # Calculate volume flow using various models for df_total
    for mo in post_prcs.DEFAULT_PREDICTOR_MODEL_FILE_NAMES:
        print(f"Post-processing model {mo} ...")
        dot_Vs_cal = []
        for index, row in df_total.iterrows():
            data_dict = {df_total.columns[i]: row[i] for i in range(1, len(row) - 1)}  # Retrieves col name, row value at index i
            dot_V_cal = post_prcs.model_single_predict(model_name=mo, features_data=data_dict)
            dot_Vs_cal.append(dot_V_cal)
        # Add result to df
        df_total[f'dot_V_{mo}'] = dot_Vs_cal
    # Save df to dir
    print(f"Saving df_total to {dir_output} ...")
    df_total.to_csv(os.path.join(dir_output, f'{case_name}_{room_name}_df_total.csv'), sep=";", index=False)

    # Split df_total to training and testing df
    df_train = df_total[(df_total['Time'] > start_time_train) & (df_total['Time'] < end_time_train)]
    df_test = df_total[(df_total['Time'] > start_time_test) & (df_total['Time'] < end_time_test)]
    dfs_filtered = {'train': df_train, 'test': df_test, 'total': df_total}

    # Analyse score, MAE, and RMSE for each model
    df_res_rows = []
    for mo in post_prcs.DEFAULT_PREDICTOR_MODEL_FILE_NAMES:
        res = {}
        for c in ['train', 'test', 'total']:
            y_act = dfs_filtered[c]['dot_V [m3/h]'].to_numpy()
            y_pred = dfs_filtered[c][f'dot_V_{mo}'].to_numpy()
            res.update({
                f'score_{c}': DataAnalyses.calculate_r_squared(y_act, y_pred),
                f'mae_{c}': DataAnalyses.calculate_mae(y_act, y_pred),
                f'rmse_{c}': DataAnalyses.calculate_rmse(y_act, y_pred),
            })
        # Read values from summary data
        res.update(**post_prcs.get_values_from_summary_data(
            model_name=mo, keys=['Feature importance', 'Computation Time']))
        # Append res to row
        df_res_rows.append({
            'model': mo, 'room': room_name,
            'score_train': res['score_train'], 'score_test': res['score_test'], 'score_total': res['score_total'],
            'mae_train': res['mae_train'], 'mae_test': res['mae_test'], 'mae_total': res['mae_total'],
            'rmse_train': res['rmse_train'], 'rmse_test': res['rmse_test'], 'rmse_total': res['rmse_total'],
            'FI': res['Feature importance'], 'CT': res['Computation Time']
        })
    # Generate result df
    df_res = pd.DataFrame(df_res_rows)
    # Save df to dir
    print(f"Saving df_res to {dir_output} ...")
    df_res.to_csv(os.path.join(dir_output, f'{case_name}_{room_name}_df_res.csv'), sep=";", index=False)


def run_auto_tuning(
        file_trial_input: str,
        dir_data_tuning_output: str = "TrialInput",
        column_of_signal: int = 7
):
    """
    file_trial_input: Input xlsx file with features and target
    dir_data_tuning_output: Data tuning output, as model tuning input
    column_of_signal: Col number of target
    """
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
    all_rooms = ['OL', 'UL', 'UR', 'OLULUR']
    for rm in all_rooms:
        SV.NameOfSubTest = f"TrialTunedModel_{rm}"
        SV.StartTraining = FILTER_PARSE[rm]['train'][0]
        SV.EndTraining = FILTER_PARSE[rm]['train'][1]
        SV.StartTesting = FILTER_PARSE[rm]['test'][0]
        SV.EndTesting = FILTER_PARSE[rm]['test'][1]
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
            room_name=rm,
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
    loop_auto_tuning(features_str="Ua.Dt.Ageo", column_of_signal=3)
    loop_auto_tuning(features_str="Ua.Dt.H.Ageo", column_of_signal=4)
    loop_auto_tuning(features_str="Ua.Dt.H.W.Ageo", column_of_signal=5)
