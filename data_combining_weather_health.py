import pandas as pd
import numpy as np
import os
import json
from geopy.geocoders import Nominatim
from datetime import datetime, timedelta


def combine_data_user_2 (raw_data_path: str="raw_data/raw_data_user_2"):

    # Search for exercise file
    exercise_data = [i for i in os.listdir(raw_data_path) if ("exercise" in i) and ("weather" not in i)][0]

    # Load exercise file
    load_data_file = lambda y: pd.read_csv(os.path.join(f"{os.getcwd()}", raw_data_path, str(y)), header=1, index_col=False)
    data_exercise = load_data_file(exercise_data)

    # Filter exercise files that contain location info
    list_of_location_data_files = [j for j in data_exercise[[i for i in data_exercise.columns if "exercise.location_data" in i][0]].tolist() if pd.notna(j)]

    # Find file names of location data
    def find_file(filename, folder):
        for root, dirs, files in os.walk(folder):
            if filename in files:
                return os.path.join(root, filename)
        return None

    location_data_files = []

    for location_data_file_name in list_of_location_data_files:
        path_of_folder = find_file(location_data_file_name, os.path.join(raw_data_path, "jsons"))
        if path_of_folder != None:
            location_data_files.append(path_of_folder)


    # Read location data & filter
    location_data = []

    for location_data_file in location_data_files:

        with open(location_data_file, "r") as f:
            data_temp = json.load(f)
            if len(data_temp) > 20:
                location_data.append(data_temp)

    location_data_updates = []
    for location_update in location_data:
        location_data_updates.append(
            {
                "start_time":location_update[0].get("start_time"),
                "latitude":location_update[0].get("latitude"),
                "longitude":location_update[0].get("longitude")
            }
        )

    # Reverse lookup city & transform time to date_time format
    location_data_updates_incl_city_names = []
    geolocator = Nominatim(user_agent="my_geocoder")

    for location_data_update in location_data_updates:
        lat = location_data_update["latitude"]
        long = location_data_update["longitude"]
        start_time = location_data_update["start_time"]

        trash, trash, trash, city, trash, trash, trash, trash = geolocator.reverse((lat, long)).address.split(", ")

        # Transform time to readable format
        dt = datetime.fromtimestamp(int(start_time)/1000)

        # Round to nearest hour
        if dt.minute >= 30:
            dt = dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            dt = dt.replace(minute=0, second=0, microsecond=0)

        formatted_dt = dt.strftime("%Y-%m-%d %H:%M:%S")

        # Save new dict
        location_data_updates_incl_city_names.append(
            {
                "time":formatted_dt,
                "city":city
            }
        )
        

    return location_data_updates_incl_city_names


temp = combine_data_user_2()
print(temp)
