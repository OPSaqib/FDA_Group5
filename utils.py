import pandas as pd
import numpy as np
import os





def transform_raw_data_user_2(folder_path: str) -> list:

    # Data file tags needed
    file_tag_heart_rate = "heart_rate"
    file_tag_step_count = "step_count"
    file_tag_step_daily_trend = "step_daily_trend"

    # Lambda function to get file name
    extract_data_file = lambda x: [i for i in os.listdir(folder_path) if str(x) in i][0]

    # Lambda function to load data
    load_data_file = lambda y: pd.read_csv(os.path.join(f"{os.getcwd()}", folder_path, str(y)), header=1, index_col=False)

    # load all data



    return None
