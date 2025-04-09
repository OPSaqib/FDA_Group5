import pandas as pd
import numpy as np
import os

# Final cleaning steps for the datasets
# Should be run as the last step

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs very basic cleaning on merged datasets
    '''
    # Split conditions into separate columns
    df_cond = df['conditions'].str.get_dummies(sep=', ')
    df = pd.concat([df, df_cond], axis=1)
    return df

for file in os.listdir('merged_weather_health_data'):
    # Load the dataset
    df = pd.read_csv(os.path.join('merged_weather_health_data', file))
    # Clean the dataset
    df_cleaned = clean_dataset(df)
    # Save the cleaned dataset
    df_cleaned.to_csv(os.path.join('cleaned_final_data', file), index=False)