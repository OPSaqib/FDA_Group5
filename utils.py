import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt

def convert_user_1(raw_data):
    return

def convert_user_2(raw_data):
    return

def convert_user_3(raw_data):
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
#convert_user_1()
#convert_user_2()
#convert_user_3()
#convert_user_4()
convert_user_5('raw_data/raw_data_user_5.csv')