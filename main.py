import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime as dt

# Load and merge heartrate datasets
df_heartrate = pd.DataFrame()
for i in range(1,6):
    try:
        # Load csv
        temp_df = pd.read_csv(f'converted_data/heart_rate_data_user_{i}.csv')
        # Standardize index
        if 'day_time' in temp_df.columns:
            temp_df.set_index('day_time')
        # Account for crucial missing column - user ID
        if 'test_subject' not in temp_df.columns:
            temp_df['test_subject'] = i
        # Merge into single dataframe
        df_heartrate = pd.concat(objs=[df_heartrate, temp_df], ignore_index=True, sort=False)
    # Handle cases of missing csv files
    except FileNotFoundError:
        print(f'Heart rate csv for user {i} not found!')


# Load and merge step count datasets
df_steps = pd.DataFrame()
for j in range(1,6):
    try:
        # Load csv
        temp_df = pd.read_csv(f'converted_data/step_count_data_user_{j}.csv')
        # Standardize index
        if 'start_time' in temp_df.columns:
            temp_df.set_index('start_time')
        # Account for crucial missing column - user ID
        if 'test_subject' not in temp_df.columns:
            temp_df['test_subject'] = j
        # Merge into single dataframe
        df_steps = pd.concat(objs=[df_steps, temp_df], ignore_index=True, sort=False)
    # Handle cases of missing csv files
    except FileNotFoundError:
        print(f'Step count csv for user {j} not found!')

# Load and merge daily step trend datasets
df_daily_steps = pd.DataFrame()
for k in range(1,6):
    try:
        # Load csv
        temp_df = pd.read_csv(f'converted_data/step_count_daily_trend_user_{k}.csv')
        # Standardize index
        if 'day_time' in temp_df.columns:
            temp_df.set_index('day_time')
        # Account for crucial missing column - user ID
        if 'test_subject' not in temp_df.columns:
            temp_df['test_subject'] = k
        # Merge into single dataframe
        df_daily_steps = pd.concat(objs=[df_daily_steps, temp_df], ignore_index=True, sort=False)
    # Handle cases of missing csv files
    except FileNotFoundError:
       print(f'Daily step trend csv for user {k} not found!')

# Additional cleaning steps
# Sort on dates for better visualizations later(?)
df_heartrate = df_heartrate.sort_values(by="date_time")
df_steps = df_steps.sort_values(by="start_time_interval")
df_daily_steps = df_daily_steps.sort_values(by="day_time")

# Some users do not have date_time, but start_time and end_time instead. Use start_time as the date_time.
df_heartrate['date_time'] = df_heartrate['date_time'].combine_first(df_heartrate['start_time'])
# Same thing applies for heart_rate & avg_heart_rate, heart_rate_max & max_heart_rate, etc.
df_heartrate['heart_rate'] = df_heartrate['heart_rate'].combine_first(df_heartrate['avg_heart_rate'])
df_heartrate['heart_rate_min'] = df_heartrate['heart_rate_min'].combine_first(df_heartrate['min_heart_rate'])
df_heartrate['heart_rate_max'] = df_heartrate['heart_rate_max'].combine_first(df_heartrate['max_heart_rate'])

# Remove an erroneous index column, and the now unneeded duplicate columns
df_heartrate = df_heartrate.drop(columns=['Unnamed: 0', 'start_time', 'end_time', 'avg_heart_rate', 'max_heart_rate', 'min_heart_rate'])
df_steps = df_steps.drop(columns=['Unnamed: 0'])
df_daily_steps = df_daily_steps.drop(columns=['Unnamed: 0'])

# Fix datetime format inconsistencies
df_heartrate['date_time'] = df_heartrate['date_time'].str.replace(r'(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', regex=True)
df_heartrate['date_time'] = pd.to_datetime(df_heartrate['date_time'])
df_steps['start_time_interval'] = df_steps['start_time_interval'].str.replace(r'(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', regex=True)
df_steps['start_time_interval'] = pd.to_datetime(df_steps['start_time_interval'])
df_steps['end_time_interval'] = df_steps['end_time_interval'].str.replace(r'(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', regex=True)
df_steps['end_time_interval'] = pd.to_datetime(df_steps['end_time_interval'])
df_daily_steps['day_time'] = df_daily_steps['day_time'].str.replace(r'(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', regex=True)
df_daily_steps['day_time'] = pd.to_datetime(df_daily_steps['day_time'])

# Filter out a row that had a datetime of January 5th 2071 - this date has not happened yet :). Using the course end date as the cutoff point
df_heartrate = df_heartrate[df_heartrate['date_time'] < dt.datetime(2030, 4, 10)]
# Also filter out data before february 2025, because this is outside our study range. Using the course start date as the cutoff point
df_heartrate = df_heartrate[df_heartrate['date_time'] > dt.datetime(2025, 2, 10)]

# TODO: more cleaning if needed
