# Input and output file paths
input_file = 'merged_health_fitness_data.csv'
heart_rate_output_file = 'heart_rate_data.csv'
daily_steps_output_file = 'daily_steps_distance.csv'

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
                    'heart_rate': bpm
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
    writer.writerow(['date_time', 'heart_rate'])
    for entry in heart_rate_data:
        writer.writerow([entry['date_time'], entry['heart_rate']])

# Write FILE 2: daily_steps_distance.csv
with open(daily_steps_output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['day_time', 'daily_step_count', 'distance_covered'])
    for day in sorted(daily_data.keys()):
        writer.writerow([day, daily_data[day]['steps'], daily_data[day]['distance']])

print(f"Files generated successfully: {heart_rate_output_file}, {daily_steps_output_file}")
