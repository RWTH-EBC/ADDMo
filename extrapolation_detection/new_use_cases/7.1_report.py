import os

import pandas

import extrapolation_detection.machine_learning_util.data_handling as dh

def create_cross_experiment_reports(use_cases: list[str], detectors: list[str], score: str):
    '''Creates csv with already available scores, just on one place for the specified use cases and
    classifiers'''

    # create empty dataframe
    df = pandas.DataFrame(columns=use_cases, index=detectors)

    # Load use cases and classifiers
    for use_case in use_cases:
        for detector_name in detectors:
            path = os.path.join(use_case, f"detector_evaluation_grid_{detector_name}.csv")
            data = pandas.read_csv(path, sep=";", encoding="unicode_escape", index_col=0)
            df.loc[detector_name, use_case] = float(data.loc[score].iloc[0])

    dh.write_csv(df, score, 'cross_experiment_results')
    print(f"Data saved to: cross_experiment_results\{score}.csv")

if __name__ == '__main__':
    use_cases = [
        "Carnot_Test2",

    ]
    detectors = [
        'KNN_val+test'
    ]
    score = 'recall' #f, fbeta, precision, recall

    create_cross_experiment_reports(use_cases, detectors, score)