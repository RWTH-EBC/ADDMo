import pandas as pd
import numpy as np
import math
import itertools
import os
import multiprocessing as mp
from tqdm import tqdm
import logging.config
from Functions.ModelPostprocessing import PostProcessing
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def get_window_height_width(room: str) -> tuple:
    if room == "OL":
        return 1.335, 1.065
    elif room == "UL":
        return 1.015, 0.775
    elif room == "UR":
        return 1.315, 1.005
    else:
        raise ValueError(f"Unknown room {room}")


def opening_width_to_angle(h: float, s: float) -> float:
    """Convert opening width to angle (rad)"""
    return 2 * math.asin(s / (2 * h))


def opening_area_geometric(h: float, w: float, s: float) -> float:
    if s < 0:
        raise ValueError(f"Invalid 's': {s}")
    if s == 0:
        return 0
    else:
        alpha = opening_width_to_angle(h=h, s=s)
        A_1 = w * s
        A_2 = 0.5 * s * h * math.cos(alpha / 2)
        return A_1 + 2 * A_2


def opening_area_projective(h: float, w: float, s: float) -> float:
    if s < 0:
        raise ValueError(f"Invalid 's': {s}")
    if s == 0:
        return 0
    else:
        alpha = opening_width_to_angle(h=h, s=s)
        A_1 = w * h * (1 - math.cos(alpha))
        A_2 = 0.5 * h * math.sin(alpha) * h * math.cos(alpha)
        return A_1 + 2 * A_2


def generate_csv_sensitivity_black_box(
        dir_output: str,
        features_str: str = "Ua.Dt.H.W.Ageo",
        rt_id_str: str = "rt0.5_id0",
        room: str = "OL",
        model_name: str = "RF",
):
    """
    Generate CSV file for sensitivity analyse of grey box model
    """
    # Settings of var spaces
    s_win_vals = np.linspace(0, 0.3, 31)  # Window opening width
    dt_vals = np.linspace(0, 30, 31)  # Temperature difference room to ambient
    u_a_vals = np.linspace(0, 10, 21)  # Wind speed

    # Loop through all combinations
    combs = list(itertools.product(s_win_vals, dt_vals, u_a_vals))  # Generate all combinations
    logger.info(f"Total combinations: {len(combs)}")
    s_win_all, dt_all, u_a_all = zip(*combs)  # Unzip all values for each parameter
    s_win_all = np.array(s_win_all)
    dt_all = np.array(dt_all)
    u_a_all = np.array(u_a_all)

    # Process result df
    df_res = pd.DataFrame({'s_win': s_win_all, 'dt': dt_all, 'u_a': u_a_all})
    h_win, w_win = get_window_height_width(room=room)

    # Load model
    prcs_obj = PostProcessing(
        dir_base=os.path.join('Results', features_str, rt_id_str),
        dir_trial_tuned_data=f'TrialTunedData',
        dir_model=f'TrialTunedModel_{room}'
    )

    # Calculate volume flow
    tqdm.pandas(desc=f'Pandas processing data {features_str} / {rt_id_str} / {room} / {model_name}')
    df_res['dot_V'] = df_res.progress_apply(
        lambda row:
        prcs_obj.model_single_predict(
            model_name=model_name,
            features_data={
                'u_a [m/s]': row['u_a'],
                'dt_ia [K]': row['dt'],
                'h_win [m]': h_win,
                'w_win [m]': w_win,
                'Ageo_win [m2]': opening_area_geometric(h=h_win, w=w_win, s=row['s_win']),
            }
        ),
        axis=1
    )

    # Export csv
    csv_file_path = os.path.join(dir_output, features_str)
    if not os.path.exists(csv_file_path):
        os.makedirs(csv_file_path)
    csv_file_name = os.path.join(csv_file_path, f"{rt_id_str}_{room}_{model_name}.csv")
    logger.info(f"Exporting CSV file to {csv_file_name} ...")
    df_res.to_csv(csv_file_name, sep=';', index=False)


def run_calculation_worker(args):
    """
    Worker function to calculate the dot_V for single setting
    :param args: tuple, containing (training ratio, ID, room, model)
    """
    rt, idx, rm, mo = args

    try:
        generate_csv_sensitivity_black_box(
            dir_output=os.path.join("Results", "Sensitivity"),
            features_str="Ua.Dt.H.W.Ageo",  # Applicable only for this feature combination
            rt_id_str=f"rt{rt}_id{idx}",
            room=rm,
            model_name=mo
        )
    except Exception as e:
        logger.error(f"Error processing rt={rt}, id={idx}, rm={rm}, mo={mo}: {e}")


def sensitivity_analyse_main(df_agg_file: str):
    df_agg = pd.read_csv(df_agg_file, sep=";")

    all_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    all_rooms = ['OL', 'UL', 'UR']
    all_models = ['ANN', 'GB', 'LASSO', 'RF', 'SVR']

    # Generate tasks based on all combinations
    tasks = []
    for rt in all_ratios:
        for rm in all_rooms:
            for mo in all_models:
                df_agg_filterd = df_agg[
                        (df_agg['features'] == "Ua.Dt.H.W.Ageo") &
                        (df_agg['model'] == mo) & (df_agg['room'] == rm) & (df_agg['train_ratio'] == rt)
                    ]
                if not df_agg_filterd.empty:
                    best_idx = df_agg_filterd['best_train_index'].values[0]
                    tasks.append((rt, best_idx, rm, mo))
                else:
                    logger.warning(f"No result with filter {rt} / {rm} / {mo}, skipping ...")
                    continue

    logger.info(f"Starting sensitivity analyse with {len(tasks)} tasks")
    for i, tsk in enumerate(tasks):
        print(f"Task {i}: rt={tsk[0]}, idx={tsk[1]}, rm={tsk[2]}, mo={tsk[3]}")

    # Determine the number of worker processes
    num_prcs = mp.cpu_count()
    logger.info(f"Number of processes: {num_prcs}")

    # Distribute tasks to workers
    with mp.Pool(processes=num_prcs) as pool:
        pool.map(run_calculation_worker, tasks)


if __name__ == '__main__':
    sensitivity_analyse_main(df_agg_file=os.path.join("Results", "PostProcessing", "df_agg.csv"))
