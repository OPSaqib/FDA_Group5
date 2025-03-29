import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime as dt
import seaborn as sns

# Main file for creating (basic) visualizations of our data

# Cleaning function
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs very basic cleaning on merged datasets
    '''
    # Filter out irrelevant columns
    df = df.drop(columns=['icon', 'stations', 'solarenergy', 'severerisk'])
    # Split conditions into separate columns
    df_cond = df['conditions'].str.get_dummies(sep=', ')
    df = pd.concat([df, df_cond], axis=1)
    return df

# Line plotting function
def plot_basic_line(df: list[pd.DataFrame], x: list[str], y: list[str], titles: list[str]) -> None:
    '''
    Produces vertically stacked basic line plots using the given dataframes.
    Expects 3 dataframes, but more or less is technically supported.
    '''
    # Tracker for amount of subplots to make
    count = len(df)
    # Initialize plot
    fig, ax = plt.subplots(count, 1, squeeze=False, figsize=(24,18))
    # Create a colormap from amount of users
    num_subjects = df[0]["test_subject"].nunique() # Use the first df to determine amount of users to map
    colors = cm.viridis(np.linspace(0, 1, num_subjects))  # Generate distinct colors
    subject_colors = {subject: color for subject, color in zip(sorted(df[0]["test_subject"].unique()), colors)}
    # Start making subplots
    for i, frame in enumerate(df):
        for subject in frame["test_subject"].unique():
            try:
                subset = frame[frame["test_subject"] == subject]
                ax[i,0].plot(subset[x[i]], subset[y[i]], c=subject_colors[subject], label=f'Subject {subject}')
            except KeyError:
                print(f"Error: subject {subject} not found in dataframe index {i}!")
        ax[i,0].set_title(titles[i])
        ax[i,0].tick_params(axis='x', labelrotation=45, labelsize=6)
        ax[i,0].legend()
        ax[i,0].xaxis.set_major_locator(plt.MaxNLocator(30))
    return None

def plot_histograms(df: list[pd.DataFrame], value: list[str], titles: list[str], bin_scale: list[int]) -> None:
    '''
    Produces histograms from the input dataframes.
    Expects EXACTLY 4 dataframes passed (can be duplicated), otherwise will likely mess up.
    '''
    # Plot layout tracking variables
    x = -1
    y = 0
    # Initialize plots
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(12,12))
    for i, frame in enumerate(df):
        row, col = divmod(i, 2) 
        sns.histplot(data=frame, x=value[i], ax=ax[row, col], kde=True, 
                     bins=int((frame[value[i]].max() - frame[value[i]].min()) / bin_scale[i]))
        ax[row, col].set_title(titles[i])
    return None

# Load datasets
df_hr_vis = pd.read_csv('merged_weather_health_data/heart_rate_data_merged_incl_weather_hourly.csv')
df_st_vis = pd.read_csv('merged_weather_health_data/step_count_data_merged_incl_weather_hourly.csv')
df_dl_vis = pd.read_csv('merged_weather_health_data/step_count_daily_trend_data_merged_incl_weather_daily.csv')
# Apply cleaning
df_hr_vis, df_st_vis, df_dl_vis = clean_dataset(df_hr_vis), clean_dataset(df_st_vis), clean_dataset(df_dl_vis)

# Aggresive NaN removal - REMOVE THIS LINE ONCE WE HAVE A WAY OF DEALING WITH NAN VALUES!
#df_hr_vis = df_hr_vis.dropna()
#df_st_vis = df_st_vis.dropna()
#df_dl_vis = df_dl_vis.dropna()

# Basic line plot call
plot_basic_line([df_hr_vis, df_st_vis, df_dl_vis], 
                ['date_time_start_hourly', 'start_time_interval_hourly', 'day_time'], 
                ['heart_rate_max','step_count','daily_step_count'],
                ['Hourly maximum heartrate over time', 'Hourly step count over time', 'Daily step count over time'])

plot_histograms([df_hr_vis, df_hr_vis, df_st_vis, df_dl_vis],
                ['heart_rate_min', 'heart_rate_max', 'step_count', 'daily_step_count'],
                ['Minimum heartrate distribution', 'Maximum heartrate distribution', 'Step count distribution', 'Daily step count distribution'],
                [3, 3, 300, 500])

plt.show() # Note that this can sometimes take a while to render out the plots - be patient!



