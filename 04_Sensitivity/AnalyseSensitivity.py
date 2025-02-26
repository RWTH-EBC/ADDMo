import pandas as pd
import numpy as np
import math
import itertools
import os
from tqdm import tqdm
import logging.config
from ModelPostprocessing import PostProcessing
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


def generate_csv_sensitivity_black_box(case: str, room: str, file_path: str):
    """
    Generate CSV file for sensitivity analyse of grey box model
    :param case: Case to analyse
    :param room: Room to apply
    :param file_path: CSV file path
    :return: Case string
    """
    # Settings of var spaces
    s_win_vals = np.linspace(0, 0.3, 61)  # Window opening width
    dt_vals = np.linspace(0, 40, 81)  # Temperature difference room to ambient
    u_a_vals = np.linspace(0, 10, 21)  # Wind speed

    # Loop through all combinations
    combs = list(itertools.product(s_win_vals, dt_vals, u_a_vals))  # Generate all combinations
    logger.info(f"Total combinations: {len(combs)}")
    s_win_all, dt_all, u_a_all = zip(*combs)  # Get all values for each parameter
    s_win_all = np.array(s_win_all)
    dt_all = np.array(dt_all)
    u_a_all = np.array(u_a_all)

    # Process result df
    df_res = pd.DataFrame({'s_win': s_win_all, 'dt': dt_all, 'u_a': u_a_all})
    h_win, w_win = get_window_height_width(room=room)

    # Load model
    dir_workspace = r"D:\sciebo\JJI_Privat\Diss\040_DynOptimazation\ADDMoResultData"
    prcs_obj = PostProcessing(
        dir_base=os.path.join(dir_workspace, f'TrialInput_{case}'),
        dir_trial_tuned_data=f'TrialTunedData_{case}',
        dir_model=f'TrialTunedModel_{case}'
    )

    # Calculate volume flow
    tqdm.pandas(desc=f'Pandas processing data {room}')
    df_res['dot_V'] = df_res.progress_apply(
        lambda row:
        prcs_obj.model_single_predict(
            model_name='RF',
            features_data={
                'u_a [m/s]': row['u_a'],
                'dt_ia [K]': row['dt'],
                'h_win [m]': h_win,
                'w_win [m]': w_win,
                'Aprj_win [m2]': opening_area_projective(h=h_win, w=w_win, s=row['s_win']),
            }
        ),
        axis=1
    )

    # Export csv
    logger.info(f"Exporting CSV file to {file_path} ...")
    df_res.to_csv(file_path, sep=';', index=False)

    return f"{case}_{room}"


if __name__ == '__main__':
    for rm in ['OL', 'UL', 'UR']:
        case = '0.7_Dyn_u.dt'
        generate_csv_sensitivity_black_box(
            case=case,
            room=rm,
            file_path=f"{case}_{rm}.csv"
        )
