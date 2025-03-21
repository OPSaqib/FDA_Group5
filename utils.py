import numpy as np
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt

def convert_user_1(raw_data):

    df = pd.read_csv(
        "raw_data_heartrate_upload.csv",
        skiprows=1,        # Skip that very first line
        usecols=range(21)  # Only read columns 0..20
    )
    #print(df.head())

    # Renameing columns of interest to simpler labels
    df = df.rename(columns={
        'com.samsung.health.heart_rate.start_time': 'start_time',
        'com.samsung.health.heart_rate.end_time': 'end_time',
        'com.samsung.health.heart_rate.heart_rate': 'avg_heart_rate',
        'com.samsung.health.heart_rate.max': 'max_heart_rate',
        'com.samsung.health.heart_rate.min': 'min_heart_rate'
    })

    #  new DataFrame with just the columns we care about
    df_hr = df[['start_time', 'end_time', 'avg_heart_rate', 'max_heart_rate', 'min_heart_rate']].copy()

    df_hr['start_time'] = pd.to_datetime(df_hr['start_time'])
    df_hr['end_time'] = pd.to_datetime(df_hr['end_time'])

    df_hr.head(30)

    df_hr.to_csv('Processed_heart_rate_data1.csv', index=False)
    
    return


def convert_user_2(folder_path: str="raw_data/raw_data_user_2") -> None:
    '''
    Transforms data from user 2 into the agreed format, and subsequently
    saves the data to the folder converted_data.
    
    Device used is a Samsung Galaxy Fit 3.

    :param folder_path: folder location where raw data is stored.
    '''
    # Data file tags needed
    file_tag_heart_rate = "heart_rate"
    file_tag_step_count = "step_count"
    file_tag_step_daily_trend = "step_daily_trend"

    # Lambda function to get file name
    extract_data_file = lambda x: [i for i in os.listdir(folder_path) if str(x) in i][0]

    # Lambda function to load data
    load_data_file = lambda y: pd.read_csv(os.path.join(f"{os.getcwd()}", folder_path, str(y)), header=1, index_col=False)

    # load all data
    heart_rate_data = load_data_file(extract_data_file(file_tag_heart_rate))
    step_count_data = load_data_file(extract_data_file(file_tag_step_count))
    step_daily_trend_data = load_data_file(extract_data_file(file_tag_step_daily_trend))

    # Remove unneeded text from column names.
    heart_rate_data.columns = heart_rate_data.columns.str.replace(r'^com\.samsung\.health\.heart_rate\.', '', regex=True)
    step_count_data.columns = step_count_data.columns.str.replace(r'^com\.samsung\.health\.step_count\.', '', regex=True)

    # Key words per dataframe
    heart_rate_key_words = ["start_time", "time_offset", "min", "max", "heart_rate"]
    step_count_key_words = ["walk_step", "distance", "speed", "calorie", "start_time", "end_time", "time_offset"]
    step_daily_trend_key_words = ["count", "distance", "speed", "calorie", "create_time"]

    # Lambda function to combine key words
    combine_key_words = lambda z: "|".join(z)

    # Filter dataframes based on key words
    heart_rate_data_filtered = heart_rate_data.loc[:, heart_rate_data.columns.str.contains(combine_key_words(heart_rate_key_words), case=False)]
    step_count_data_filtered = step_count_data.loc[:, step_count_data.columns.str.contains(combine_key_words(step_count_key_words), case=False)]
    step_daily_trend_data_filtered = step_daily_trend_data.loc[:, step_daily_trend_data.columns.str.contains(combine_key_words(step_daily_trend_key_words), case=False)]

    # Final adjustments heart rate data
        # Rename column names
    heart_rate_column_map = {
        "start_time":"date_time",
        "min":"heart_rate_min",
        "max":"heart_rate_max"
    }
    heart_rate_data_filtered = heart_rate_data_filtered.rename(columns=heart_rate_column_map)

        # Filter date_time to correct format
    heart_rate_data_filtered["date_time"] = pd.to_datetime(heart_rate_data_filtered["date_time"]).dt.strftime("%Y:%m:%d %H:%M:%S")
    
        # Add test subject column
    heart_rate_data_filtered["test_subject"] = 2

        # Order columns
    heart_rate_data_filtered = heart_rate_data_filtered[["heart_rate", "heart_rate_min", "heart_rate_max", "date_time", "time_offset", "test_subject"]]

    # Final adjustments step count data
        # Rename column names
    step_count_column_map = {
        "walk_step":"step_count",
        "distance":"distance_covered",
        "calorie":"calories_burned",
        "start_time":"start_time_interval",
        "end_time":"end_time_interval"
    }
    step_count_data_filtered = step_count_data_filtered.rename(columns=step_count_column_map)

        # Filter start & end time to correct format
    step_count_data_filtered["start_time_interval"] = pd.to_datetime(step_count_data_filtered["start_time_interval"]).dt.strftime("%Y:%m:%d %H:%M:%S")
    step_count_data_filtered["end_time_interval"] = pd.to_datetime(step_count_data_filtered["end_time_interval"]).dt.strftime("%Y:%m:%d %H:%M:%S")

        # step_count to float
    step_count_data_filtered["step_count"] = step_count_data_filtered["step_count"].astype(float)

        # Add test subject column
    step_count_data_filtered["test_subject"] = 2

        # Order columns
    step_count_data_filtered = step_count_data_filtered[["step_count", "distance_covered", "speed", "calories_burned", "start_time_interval", "end_time_interval", "time_offset", "test_subject"]]


    # Final adjustments step daily trend data
        # Rename column names
    step_daily_trend_column_map = {
        "create_time":"day_time",
        "count":"daily_step_count",
        "distance":"distance_covered",
        "calorie":"calories_burned"
    }
    step_daily_trend_data_filtered = step_daily_trend_data_filtered.rename(columns=step_daily_trend_column_map)

        # Filter day time to correct format
    step_daily_trend_data_filtered["day_time"] = pd.to_datetime(step_daily_trend_data_filtered["day_time"]).dt.strftime("%Y:%m:%d")

        # daily step count to float
    step_daily_trend_data_filtered["daily_step_count"] = step_daily_trend_data_filtered["daily_step_count"].astype(float)

        # Add test subject column
    step_daily_trend_data_filtered["test_subject"] = 2

        # Order columns
    step_daily_trend_data_filtered = step_daily_trend_data_filtered[["daily_step_count", "distance_covered", "speed", "calories_burned", "day_time", "test_subject"]]
    
    # Save data frames
    heart_rate_data_filtered.to_csv('converted_data/heart_rate_data_user_2.csv')
    step_count_data_filtered.to_csv('converted_data/step_count_data_user_2.csv')
    step_daily_trend_data_filtered.to_csv('converted_data/step_count_daily_trend_user_2.csv')

    return None


def convert_user_3(raw_data_path: str = "raw_data_user3.csv")

    # Load the raw heart rate CSV exported from Fitbit processing
    df = pd.read_csv(
        raw_data_path,
        skiprows=0  # No need to skip header rows as data is already clean
    )

    # Rename columns to match group standard
    df = df.rename(columns={
        'heart_rate': 'heart_rate',
        'heart_rate_min': 'heart_rate_min',
        'heart_rate_max': 'heart_rate_max',
        'date_time': 'date_time',
        'time_offset': 'time_offset',
        'test_subject': 'test_subject'
    })

    # Ensure correct datetime format
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Save the cleaned data
    df.to_csv('Processed_heart_rate_data_user3.csv', index=False)

    return


def convert_user_4(raw_data):
    return


def convert_user_5(raw_data):
    '''Function to convert raw data for user 5, device: MiBand 7'''
    # Load dataset from csv
    df = pd.read_csv(raw_data)
    # Flatten 'Value' column, which has dictionaries as values, into columns for each dictionary value
    df['Value'] = df['Value'].apply(ast.literal_eval)
    df_flat = df.drop(columns=['Value']).join(pd.DataFrame.from_records(df['Value']))
    # Remove columns if all values in it are NaN
    df_flat = df_flat.dropna(axis=1, how='all')
    # Remove PAI columns (these are not relevant to us), as well as duplicate time column
    df_flat = df_flat[df_flat['Key'] != "pai"].drop(columns=['high_zone_pai','low_zone_pai','medium_zone_pai','daily_pai','total_pai','time'])
    # Fix time columns to be more readable
    df_flat['Time'] = pd.to_datetime(df['Time'], unit='s')
    df_flat['UpdateTime'] = pd.to_datetime(df['UpdateTime'], unit='s')
    # Separate df's for each possibly relevant category - remaining unused categories are 'dynamic', 'intensity', 'valid_stand', and 'weight'.
    # Columns that do not apply to the category are also dropped (can be undone later if we need multiple categories in one df)
    df_heartrate = df_flat[df_flat['Key'] == "heart_rate"].drop(columns=['steps','calories','spo2','distance','end_time','start_time','type','date_time','weight'])
    df_steps = df_flat[df_flat['Key'] == "steps"].drop(columns=['bpm','calories','spo2','weight','type','date_time'])
    df_calories = df_flat[df_flat['Key'] == "calories"].drop(columns=['bpm','steps','spo2','weight','distance','end_time','start_time','type','date_time'])
    df_spo2 = df_flat[df_flat['Key'] == "single_spo2"].drop(columns=['bpm','steps','calories','weight','distance','end_time','start_time','type','date_time'])
    # Standardize columns
    # Heartrate
    df_hr_std = df_heartrate.rename(columns={'bpm': 'heart_rate', 'Time': 'date_time'}).drop(columns=['Key', 'UpdateTime', 'Uid', 'Sid'])
    df_hr_std['heart_rate_min'], df_hr_std['heart_rate_max'], df_hr_std['time_offset'], df_hr_std['test_subject'] = np.nan, np.nan, 'UTC+0100', 5
    # Steps
    df_st_std = df_steps.rename(columns={'steps': 'step_count', 'distance': 'distance_covered', 'Time': 'start_time_interval', 'end_time': 'end_time_interval'}).drop(columns=['Key', 'UpdateTime', 'Uid', 'Sid', 'start_time'])
    df_st_std['speed'], df_st_std['calories_burned'], df_st_std['time_offset'], df_st_std['test_subject'] = np.nan, np.nan, 'UTC+0100', 5
    # Step trend
    df_dd_std = df_st_std.rename(columns={'start_time_interval': 'day_time', 'step_count': 'daily_step_count'}).drop(columns=['end_time_interval'])
    df_dd_std['day_time'] = df_dd_std['day_time'].apply(lambda x: x.date())
    df_dd_std = df_dd_std.groupby('day_time', as_index=True).sum()
    df_dd_std['test_subject'] = 5
    df_dd_std[['speed', 'calories_burned']] = df_dd_std[['speed', 'calories_burned']].replace(0.0, np.nan, inplace=True)
    # Save results to new csv files
    df_hr_std.to_csv('converted_data/heart_rate_data_user_5.csv')
    df_st_std.to_csv('converted_data/step_count_data_user_5.csv')
    df_dd_std.to_csv('converted_data/step_count_daily_trend_user_5.csv')
    return


# Execute functions
# convert_user_1()
# convert_user_2()
# convert_user_3()
# convert_user_4()
# convert_user_5('raw_data/raw_data_user_5.csv')
