import numpy as np
import pandas as pd
import ast
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


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

        # Add location 
    heart_rate_data_filtered["date_time_parsed"] = pd.to_datetime(
        heart_rate_data_filtered["date_time"], format="%Y:%m:%d %H:%M:%S"
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
    step_count_data_filtered["start_time_interval"] = pd.to_datetime(step_count_data_filtered["start_time_interval"]).dt.strftime("%Y:%m:%d %H:%M:%S")
    step_count_data_filtered["end_time_interval"] = pd.to_datetime(step_count_data_filtered["end_time_interval"]).dt.strftime("%Y:%m:%d %H:%M:%S")

        # step_count to float
    step_count_data_filtered["step_count"] = step_count_data_filtered["step_count"].astype(float)

        # Add test subject column
    step_count_data_filtered["test_subject"] = 2

        # Add location 
    step_count_data_filtered["date_time_parsed"] = pd.to_datetime(
        step_count_data_filtered["start_time_interval"], format="%Y:%m:%d %H:%M:%S"
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
    step_daily_trend_data_filtered["day_time"] = pd.to_datetime(step_daily_trend_data_filtered["day_time"]).dt.strftime("%Y:%m:%d")

        # daily step count to float
    step_daily_trend_data_filtered["daily_step_count"] = step_daily_trend_data_filtered["daily_step_count"].astype(float)

        # Add test subject column
    step_daily_trend_data_filtered["test_subject"] = 2

        # Add location 
    step_daily_trend_data_filtered["date_time_parsed"] = pd.to_datetime(
        step_daily_trend_data_filtered["day_time"], format="%Y:%m:%d"
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


def convert_user_4(raw_data):
    return


def convert_user_5(raw_data: str) -> None:
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


# Execute functions
# convert_user_1()
# convert_user_2()
# convert_user_3()
# convert_user_4()
# convert_user_5('raw_data/raw_data_user_5.csv')


def merge_users_data() -> None:
    heart_rate_data = [pd.read_csv(f"converted_data/heart_rate_data_user_{i}.csv") for i in range(1,6)]
    step_count_daily_trend_data = [pd.read_csv(f"converted_data/step_count_daily_trend_data_user_{i}.csv") for i in range(1,6)]
    step_count_data = [pd.read_csv(f"converted_data/step_count_data_user_{i}.csv") for i in range(1,6)]

    # Concatenate function
    concat_data = lambda x: pd.concat(x, axis=0, ignore_index=True)

    heart_rate_concat_data = concat_data(heart_rate_data).sort_values(by=["test_subject", "date_time"])
    step_count_daily_trend_concat_data = concat_data(step_count_daily_trend_data).sort_values(by=["test_subject", "day_time"])
    step_count_concat_data = concat_data(step_count_data).sort_values(by=["test_subject", "start_time_interval"])

    heart_rate_concat_data.to_csv("merged_data/heart_rate_data_merged.csv")
    step_count_daily_trend_concat_data.to_csv("merged_data/step_count_daily_trend_data_merged.csv")
    step_count_concat_data.to_csv("merged_data/step_count_data_merged.csv")
    
    return None
    

def map_weather_to_health_data(file_path: str) -> None:
    # Load weather data
    weather_data = pd.concat([
        pd.read_csv("weather_data/Weather Data Hourly 2025-02-10 to 2025-02-28.csv"),
        pd.read_csv("weather_data/Weather Data Hourly 2025-03-01 to 2025-03-16.csv"),
        pd.read_csv("weather_data/Weather Data Hourly 2025-03-17 to 2025-03-30.csv")
    ], axis=0, ignore_index=True).sort_values(by=["name", "datetime"]).reset_index(drop=True)

    # Convert ISO format to datetime string
    weather_data["datetime"] = pd.to_datetime(weather_data["datetime"]).dt.strftime("%Y:%m:%d %H:%M:%S")

    # Load and parse health data
    health_data = pd.read_csv(file_path)
    if "Unnamed: 0" in health_data.columns:
        health_data = health_data.drop(columns=["Unnamed: 0"])

    # Transform date_time column to datetime format
    health_data["date_time"] = pd.to_datetime(health_data["date_time"], format="%Y:%m:%d %H:%M:%S")

    # Filter time range
    start = datetime.strptime("2025:02:10 00:00:00", "%Y:%m:%d %H:%M:%S")
    end = datetime.strptime("2025:03:30 23:59:00", "%Y:%m:%d %H:%M:%S")
    mask = (health_data["date_time"] >= start) & (health_data["date_time"] <= end)
    filtered_health_data = health_data.loc[mask].copy()

    # Convert date_time format back to string
    filtered_health_data["date_time"] = filtered_health_data["date_time"].dt.strftime("%Y:%m:%d %H:%M:%S")

    # Round to nearest hour
    def determine_nearest_hour(x):
        dt = datetime.strptime(x, "%Y:%m:%d %H:%M:%S")
        if dt.minute >= 30:
            return (dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)).strftime("%Y:%m:%d %H:%M:%S")
        else:
            return dt.replace(minute=0, second=0, microsecond=0).strftime("%Y:%m:%d %H:%M:%S")

    filtered_health_data["date_time_w_rounded_hour"] = filtered_health_data["date_time"].apply(determine_nearest_hour)

    # Add default location if missing
    if "location" not in filtered_health_data.columns:
        filtered_health_data.loc[:, "location"] = "Eindhoven"

    # Merge with weather data
    merged_data = pd.merge(
        filtered_health_data,
        weather_data,
        how="left",
        left_on=["location", "date_time_w_rounded_hour"],
        right_on=["name", "datetime"]
    )

    merged_data.to_csv(f"merged_weather_health_data/{file_path.split("/")[1].replace(".csv", "")}_incl_weather.csv")

    return None


# merge_users_data()
# map_weather_to_health_data("merged_data/heart_rate_data_merged.csv")
# map_weather_to_health_data("merged_data/step_count_daily_trend_data_merged.csv")
# map_weather_to_health_data("merged_data/step_count_data_merged.csv")
