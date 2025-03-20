import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and merge heartrate datasets
df_heartrate = pd.DataFrame()
for i in range(1,6):
    try:
        # Load csv
        temp_df = pd.read_csv(f'converted_data/heart_rate_data_user_{i}.csv')
        # Standardize index
        if 'day_time' in temp_df.columns:
            temp_df.set_index('day_time')
        # Merge into single dataframe
        df_heartrate = pd.concat(objs=[df_heartrate, temp_df], ignore_index=True, sort=False)
    # Handle cases of missing csv files
    except FileNotFoundError:
        print(f'Heart rate csv for user {i} not found!')

# Load and merge step count datasets
df_steps = pd.DataFrame()
for i in range(1,6):
    try:
        # Load csv
        temp_df = pd.read_csv(f'converted_data/step_count_data_user_{i}.csv')
        # Standardize index
        if 'start_time' in temp_df.columns:
            temp_df.set_index('start_time')
        # Merge into single dataframe
        df_steps = pd.concat(objs=[df_steps, temp_df], ignore_index=True, sort=False)
    # Handle cases of missing csv files
    except FileNotFoundError:
        print(f'Step count csv for user {i} not found!')

# Load and merge daily step trend datasets
df_daily_steps = pd.DataFrame()
for i in range(1,6):
    try:
        # Load csv
        temp_df = pd.read_csv(f'converted_data/step_count_daily_trend_user_{i}.csv')
        # Standardize index
        if 'day_time' in temp_df.columns:
            temp_df.set_index('day_time')
        # Merge into single dataframe
        df_daily_steps = pd.concat(objs=[df_daily_steps, temp_df], ignore_index=True, sort=False)
    # Handle cases of missing csv files
    except FileNotFoundError:
        print(f'Daily step trend csv for user {i} not found!')

print(df_heartrate)
print(df_steps)
print(df_daily_steps)