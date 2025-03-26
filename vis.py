import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime as dt

# Main file for creating (basic) visualizations of our data

# Cleaning function
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Filter out irrelevant columns
    df = df.drop(columns=['icon', 'stations', 'solarenergy', 'severerisk'])
    # Split conditions into separate columns
    df_cond = df['conditions'].str.get_dummies(sep=', ')
    df = pd.concat([df, df_cond], axis=1)
    return df

# Load datasets
df_hr_vis = pd.read_csv('merged_weather_health_data/heart_rate_data_merged_incl_weather_hourly.csv')
df_st_vis = pd.read_csv('merged_weather_health_data/step_count_data_merged_incl_weather_hourly.csv')
df_dl_vis = pd.read_csv('merged_weather_health_data/step_count_daily_trend_data_merged_incl_weather_daily.csv')
# Apply cleaning
df_hr_vis, df_st_vis, df_dl_vis = clean_dataset(df_hr_vis), clean_dataset(df_st_vis), clean_dataset(df_dl_vis)

# Basic line visualizations
fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(12,12))

# Define a colormap
num_subjects = df_hr_vis["test_subject"].nunique()
colors = cm.viridis(np.linspace(0, 1, num_subjects))  # Generate distinct colors

# Create a mapping from test_subject to colors
subject_colors = {subject: color for subject, color in zip(sorted(df_hr_vis["test_subject"].unique()), colors)}

# Heart Rate Plot
for subject in df_hr_vis["test_subject"].unique():
    subset = df_hr_vis[df_hr_vis["test_subject"] == subject]
    ax[0,0].step(subset["date_time_start_hourly"], subset["heart_rate_max"], c=subject_colors[subject], label=f'Subject {subject}')
ax[0,0].set_title('Heartrate (bpm)')
ax[0,0].tick_params(axis='x', labelrotation=45)
ax[0,0].legend()

# Steps Plot
for subject in df_st_vis["test_subject"].unique():
    subset = df_st_vis[df_st_vis["test_subject"] == subject]
    ax[1,0].scatter(subset["start_time_interval"], subset["step_count"], color=subject_colors[subject], label=f'Subject {subject}')
ax[1,0].set_title('Steps')
ax[1,0].tick_params(axis='x', labelrotation=45)
ax[1,0].legend()

# Daily Steps Plot
for subject in df_dl_vis["test_subject"].unique():
    subset = df_dl_vis[df_dl_vis["test_subject"] == subject]
    ax[2,0].scatter(subset["day_time"], subset["daily_step_count"], color=subject_colors[subject], label=f'Subject {subject}')
ax[2,0].set_title('Daily steps')
ax[2,0].tick_params(axis='x', labelrotation=45)
ax[2,0].legend()

#plt.show()

fig2, ax = plt.subplots(2, 2, squeeze=False, figsize=(12,12))
ax[0,0].hist(df_hr_vis['heart_rate_min'], bins=int((df_hr_vis['heart_rate_min'].max()-df_hr_vis['heart_rate_min'].min())/2))
ax[0,0].set_title('Minimum heartrate distribution')

ax[0,1].hist(df_hr_vis['heart_rate_max'], bins=int((df_hr_vis['heart_rate_max'].max()-df_hr_vis['heart_rate_max'].min())/2))
ax[0,1].set_title('Maximum heartrate distribution')

ax[1,0].hist(df_st_vis['step_count'], bins=int((df_st_vis['step_count'].max()-df_st_vis['step_count'].min())/2))
ax[1,0].set_title('Step count distribution')

ax[1,1].hist(df_dl_vis['daily_step_count'], bins=int((df_dl_vis['daily_step_count'].max()-df_dl_vis['daily_step_count'].min())/2))
ax[1,1].set_title('Daily step count distribution')

plt.show()
