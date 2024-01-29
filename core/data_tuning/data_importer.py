import pandas as pd


def load_data(abs_path: str) -> pd.DataFrame: #todo: move to data_handling or load_save

    if abs_path.endswith(".csv"):
        # Read the CSV file
        df = pd.read_csv(abs_path, delimiter=';', index_col=[0], encoding='latin1', header=[0])
    elif abs_path.endswith(".xlsx"):
        # Read the Excel file
        df = pd.read_excel(abs_path, index_col=[0], header=[0])

    # Convert the index to datetime
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M %Z')

    return df

