import numpy as np
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import csv
import json
from collections import defaultdict
import pytz
import math


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
    heart_rate_data_filtered["date_time"] = pd.to_datetime(heart_rate_data_filtered["date_time"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    
        # Add test subject column
    heart_rate_data_filtered["test_subject"] = 2

        # Add location 
    heart_rate_data_filtered["date_time_parsed"] = pd.to_datetime(
        heart_rate_data_filtered["date_time"], format="%Y-%m-%d %H:%M:%S"
    )

    start = datetime(2025, 3, 4)
    end = datetime(2025, 3, 7)

    heart_rate_data_filtered["location"] = heart_rate_data_filtered["date_time_parsed"].apply(
        lambda dt: "Torremolinos" if start <= dt <= end else "Eindhoven"
    )

    heart_rate_data_filtered = heart_rate_data_filtered.drop(columns=["date_time_parsed"])

        # Order columns
    heart_rate_data_filtered = heart_rate_data_filtered[["heart_rate", "heart_rate_min", "heart_rate_max", "date_time", "time_offset", "test_subject", "location"]]

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
    step_count_data_filtered["start_time_interval"] = pd.to_datetime(step_count_data_filtered["start_time_interval"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    step_count_data_filtered["end_time_interval"] = pd.to_datetime(step_count_data_filtered["end_time_interval"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # step_count to float
    step_count_data_filtered["step_count"] = step_count_data_filtered["step_count"].astype(float)

        # Add test subject column
    step_count_data_filtered["test_subject"] = 2

        # Add location 
    step_count_data_filtered["date_time_parsed"] = pd.to_datetime(
        step_count_data_filtered["start_time_interval"], format="%Y-%m-%d %H:%M:%S"
    )

    start = datetime(2025, 3, 4)
    end = datetime(2025, 3, 7)

    step_count_data_filtered["location"] = step_count_data_filtered["date_time_parsed"].apply(
        lambda dt: "Torremolinos" if start <= dt <= end else "Eindhoven"
    )

    step_count_data_filtered = step_count_data_filtered.drop(columns=["date_time_parsed"])

        # Order columns
    step_count_data_filtered = step_count_data_filtered[["step_count", "distance_covered", "speed", "calories_burned", "start_time_interval", "end_time_interval", "time_offset", "test_subject", "location"]]


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
    step_daily_trend_data_filtered["day_time"] = pd.to_datetime(step_daily_trend_data_filtered["day_time"]).dt.strftime("%Y-%m-%d")

        # daily step count to float
    step_daily_trend_data_filtered["daily_step_count"] = step_daily_trend_data_filtered["daily_step_count"].astype(float)

        # Add test subject column
    step_daily_trend_data_filtered["test_subject"] = 2

        # Add location 
    step_daily_trend_data_filtered["date_time_parsed"] = pd.to_datetime(
        step_daily_trend_data_filtered["day_time"], format="%Y-%m-%d"
    )

    start = datetime(2025, 3, 4)
    end = datetime(2025, 3, 7)

    step_daily_trend_data_filtered["location"] = step_daily_trend_data_filtered["date_time_parsed"].apply(
        lambda dt: "Torremolinos" if start <= dt <= end else "Eindhoven"
    )

    step_daily_trend_data_filtered = step_daily_trend_data_filtered.drop(columns=["date_time_parsed"])

        # Order columns
    step_daily_trend_data_filtered = step_daily_trend_data_filtered[["daily_step_count", "distance_covered", "speed", "calories_burned", "day_time", "test_subject", "location"]]
    
    # Save data frames
    heart_rate_data_filtered.to_csv('converted_data/heart_rate_data_user_2.csv')
    step_count_data_filtered.to_csv('converted_data/step_count_data_user_2.csv')
    step_daily_trend_data_filtered.to_csv('converted_data/step_count_daily_trend_user_2.csv')

    return None


def convert_user_3(raw_data_path: str = "raw_data_user3.csv"):

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


def convert_user_4(raw_data_folder: str= "raw_data"):
    # Input and output file paths
    input_file = f'{raw_data_folder}/raw_data_user_4.csv'  # RAW DATA FILE

    # FILES TO OUTPUT:
    heart_rate_output_file = 'converted_data/heart_rate_data_user_4.csv'
    daily_steps_output_file = 'converted_data/step_count_daily_trend_user_4.csv'
    step_count_data_file = 'converted_data/step_count_data_user_4.csv'

    # For FILE 1: Heart Rate Data
    heart_rate_data = []

    # For FILE 2: Daily Steps and Distance
    daily_data = defaultdict(lambda: {'steps': 0, 'distance': 0})

    # Read the input CSV file
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) 
        
        for row in reader:
            if not row or len(row) < 6:
                continue
            
            uid, sid, key, time, value, update_time = row
            
            try:
                # Parse the JSON value
                value_dict = json.loads(value)
                
                # Determine the correct timestamp field based on the key
                if key in ['heart_rate', 'steps', 'calories', 'intensity', 'spo2']:
                    timestamp = int(value_dict['time'])
                elif key in ['valid_stand']:
                    timestamp = int(value_dict['start_time'])  # Use start_time for valid_stand
                elif key in ['vitality', 'resting_heart_rate']:
                    timestamp = int(value_dict['date_time'])  # Use date_time for vitality/resting_heart_rate
                else:
                    continue  # Skip rows we don't need (e.g., valid_stand, vitality)

                dt = datetime.fromtimestamp(timestamp)
                
                if key == 'heart_rate':
                    # Process heart rate data
                    bpm = value_dict['bpm']
                    heart_rate_data.append({
                        'date_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                        'heart_rate': bpm,
                        'heart_rate_min': math.nan,
                        'heart_rate_max': math.nan,
                        'time_offset': 'UTC+0100',
                        'test_subject': 4,
                        'location': 'Eindhoven'
                    })
                
                elif key == 'steps':
                    # Process steps and distance data
                    day = dt.strftime('%Y-%m-%d')
                    daily_data[day]['steps'] += value_dict['steps']
                    daily_data[day]['distance'] += value_dict['distance']
                    
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"Error processing row: {row}. Error: {e}")
                continue

    # Write FILE 1: heart_rate_data.csv
    with open(heart_rate_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['date_time', 'heart_rate', 'heart_rate_min', 'heart_rate_max', 'time_offset', 'test_subject', 'location'])
        for entry in heart_rate_data:
            writer.writerow([entry['date_time'], entry['heart_rate'], entry['heart_rate_min'], entry['heart_rate_max'], entry['time_offset'], entry['test_subject'], entry['location']])

    # Write FILE 2: daily_steps_distance.csv
    with open(daily_steps_output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['day_time', 'daily_step_count', 'distance_covered', 'speed', 'calories_burned', 'test_subject', 'location'])
        for day in sorted(daily_data.keys()):
            writer.writerow([day, daily_data[day]['steps'], daily_data[day]['distance'], math.nan, math.nan, 4, 'Eindhoven'])

    print(f"Files generated successfully: {heart_rate_output_file}, {daily_steps_output_file}")

    # FOR DAILY STEPS:
    
    # Read the input CSV
    df = pd.read_csv(input_file)

    # Filter rows where Key is "steps" since we need step_count and distance_covered
    steps_df = df[df['Key'] == 'steps'].copy()

    # Function to parse the JSON in the 'Value' column
    def parse_value(json_str):
        try:
            data = json.loads(json_str)
            return pd.Series({
                'step_count': data.get('steps', float('nan')),
                'distance_covered': data.get('distance', float('nan'))
            })
        except json.JSONDecodeError:
            return pd.Series({
                'step_count': float('nan'),
                'distance_covered': float('nan')
            })

    # Function to convert epoch time to human-readable time with UTC+0100
    def epoch_to_human_readable(epoch_time):
        # Convert epoch to UTC datetime
        utc_time = datetime.fromtimestamp(epoch_time, tz=pytz.UTC)
        # Define UTC+0100 timezone
        utc_plus_1 = pytz.timezone('Etc/GMT-1')  # GMT-1 is equivalent to UTC+0100
        # Localize to UTC+0100
        local_time = utc_time.astimezone(utc_plus_1)
        # Format as string
        return local_time.strftime('%Y-%m-%d %H:%M:%S')

    # Apply the parsing function to the 'Value' column
    parsed_values = steps_df['Value'].apply(parse_value)
    steps_df = pd.concat([steps_df, parsed_values], axis=1)

    # Convert epoch time to human-readable time
    steps_df['start_time_interval'] = steps_df['Time'].apply(epoch_to_human_readable)

    # Create the output DataFrame with the required columns
    output_df = pd.DataFrame({
        'step_count': steps_df['step_count'],
        'distance_covered': steps_df['distance_covered'],
        'speed': float('nan'),  # Filled with NaN
        'calories_burned': float('nan'),  # Filled with NaN as requested
        'start_time_interval': steps_df['start_time_interval'],
        'end_time_interval': float('nan'),  # Filled with NaN
        'time_offset': 'UTC+0100',  # Time offset specified
        'test_subject': 4  # Filled with 4
    })

    # Write the output to a new CSV file
    output_df.to_csv(step_count_data_file, index=False)

    print(f"Transformation complete. Output saved to {step_count_data_file}")


def convert_user_5(raw_data: str = 'raw_data/raw_data_user_5.csv') -> None:
    '''
    Function to convert raw data for user 5, device: MiBand 7
    '''
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


def merge_users_data() -> None:
    heart_rate_data = [pd.read_csv(f"converted_data/heart_rate_data_user_{i}.csv") for i in range(1,6)]
    step_count_daily_trend_data = [pd.read_csv(f"converted_data/step_count_daily_trend_user_{i}.csv") for i in range(1,6)]
    step_count_data = [pd.read_csv(f"converted_data/step_count_data_user_{i}.csv") for i in range(1,6)]


    ## Apply some filters
    # Check if column location exists
    def add_location_to_df(df: pd.DataFrame) -> pd.DataFrame:
        if "location" not in df.columns:
            df["location"] = "Eindhoven"

        return df
    

    # Check if column "Unnamed: 0" in columns
    def remove_redundant_col(df: pd.DataFrame) -> pd.DataFrame:
        if "Unnamed: 0" in df.columns:
            df.drop(columns=['Unnamed: 0'], inplace=True)

        return df
    

    # Check if column "time_offset" in coluumns for step_count_daily_trend.
    def remove_time_offset_from_step_count_daily_trend(df: pd.DataFrame) -> pd.DataFrame:
        if "time_offset" in df.columns:
            df.drop(columns=["time_offset"], inplace=True)

        return df


    # Apply filters
    heart_rate_data = [remove_redundant_col(add_location_to_df(df)) for df in heart_rate_data]
    step_count_daily_trend_data = [remove_time_offset_from_step_count_daily_trend(remove_redundant_col(add_location_to_df(df))) for df in step_count_daily_trend_data]
    step_count_data = [remove_redundant_col(add_location_to_df(df)) for df in step_count_data]
    

    # Concatenate function
    concat_data = lambda x: pd.concat(x, axis=0, ignore_index=True)

    heart_rate_concat_data = concat_data(heart_rate_data).sort_values(by=["test_subject", "date_time"])
    step_count_daily_trend_concat_data = concat_data(step_count_daily_trend_data).sort_values(by=["test_subject", "day_time"])
    step_count_concat_data = concat_data(step_count_data).sort_values(by=["test_subject", "start_time_interval"])

    if not os.path.exists("merged_data"):
        os.makedirs("merged_data")

    heart_rate_concat_data.to_csv("merged_data/heart_rate_data_merged.csv")
    step_count_daily_trend_concat_data.to_csv("merged_data/step_count_daily_trend_merged.csv")
    step_count_concat_data.to_csv("merged_data/step_count_data_merged.csv")
    
    return None
    

def map_weather_to_heart_rate_data_hourly(file_path: str) -> None:
    # Load hourly weather data
    weather_data = pd.concat([
        pd.read_csv("weather_data/Weather Data Hourly 2025-02-10 to 2025-02-28.csv", index_col=False),
        pd.read_csv("weather_data/Weather Data Hourly 2025-03-01 to 2025-03-16.csv", index_col=False),
        pd.read_csv("weather_data/Weather Data Hourly 2025-03-17 to 2025-03-30.csv", index_col=False)
    ], axis=0, ignore_index=True).sort_values(by=["name", "datetime"]).reset_index(drop=True)

    # Convert ISO format to datetime string
    weather_data["datetime"] = pd.to_datetime(weather_data["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Load and parse health data
    health_data = pd.read_csv(file_path)
    if "Unnamed: 0" in health_data.columns:
        health_data = health_data.drop(columns=["Unnamed: 0"])

    # Transform date_time column to datetime format
    health_data["date_time"] = pd.to_datetime(health_data["date_time"], format="%Y-%m-%d %H:%M:%S")

    # Filter time range
    start = datetime.strptime("2025-02-10 00:00:00", "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime("2025-03-30 23:59:00", "%Y-%m-%d %H:%M:%S")
    mask = (health_data["date_time"] >= start) & (health_data["date_time"] <= end)
    filtered_health_data = health_data.loc[mask].copy()

    # Convert date_time format back to string
    filtered_health_data["date_time"] = filtered_health_data["date_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Round to nearest hour
    def determine_nearest_hour(x):
        dt = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        if dt.minute >= 30:
            return (dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        else:
            return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:%S")

    filtered_health_data["date_time_w_rounded_hour"] = filtered_health_data["date_time"].apply(determine_nearest_hour)

    # Merge with weather data
    merged_data = pd.merge(
        filtered_health_data,
        weather_data,
        how="left",
        left_on=["location", "date_time_w_rounded_hour"],
        right_on=["name", "datetime"]
    )

    if not os.path.exists("merged_weather_health_data"):
        os.makedirs("merged_weather_health_data")

    merged_data.to_csv("merged_weather_health_data/step_count_data_merged_incl_weather_hourly.csv")

    return None


def map_weather_to_step_count_data_hourly(file_path: str) -> None:
    # Load hourly weather data
    weather_data = pd.concat([
        pd.read_csv("weather_data/Weather Data Hourly 2025-02-10 to 2025-02-28.csv", index_col=False),
        pd.read_csv("weather_data/Weather Data Hourly 2025-03-01 to 2025-03-16.csv", index_col=False),
        pd.read_csv("weather_data/Weather Data Hourly 2025-03-17 to 2025-03-30.csv", index_col=False)
    ], axis=0, ignore_index=True).sort_values(by=["name", "datetime"]).reset_index(drop=True)

    # Convert ISO format to datetime string
    weather_data["datetime"] = pd.to_datetime(weather_data["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")



    return None

def map_weather_to_heart_rate_data_daily(file_path: str) -> None:
    pass


def map_weather_to_step_count_daily_trend_data_daily(file_path: str) -> None:
    pass


def map_weather_to_step_count_data_daily(file_path: str) -> None:
    pass


## Execute functions
# convert_user_1()
# convert_user_2()
# convert_user_3()
# convert_user_4()
# convert_user_5()

# merge_users_data()

# map_weather_to_heart_rate_data_hourly("merged_data/heart_rate_data_merged.csv")
# map_weather_to_step_count_data_hourly("merged_data/step_count_data_merged.csv")

# map_weather_to_heart_rate_data_daily("merged_data/heart_rate_data_merged.csv")
# map_weather_to_step_count_daily_trend_data_daily("merged_data/step_count_daily_trend_merged.csv")
# map_weather_to_step_count_data_daily("merged_data/step_count_data_merged.csv")

