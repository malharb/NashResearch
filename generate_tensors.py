
import pandas as pd
import numpy as np
import torch
from datetime import timedelta

# Constants
num_persons = 10
num_days = 14
num_minutes = 1440
num_samples = 1000  # Define the number of samples
d_model = 8  # Assuming dimensions 0-3 are used for other metrics, and 4-7 for sleep stages
entries_per_day_per_person = num_minutes
start_date = pd.to_datetime('2021-01-01')

# Initialize lists to store the generated data
heart_rate_entries = []
step_entries = []
activity_summary_entries = []
sleep_entries = []

# Generate random data for heart rate, step, activity summary, and sleep
np.random.seed(42)

for person_id in range(1, num_persons + 1):
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        for minute in range(entries_per_day_per_person):
            # Simulate higher frequency of heart rate and step data
            if np.random.rand() < 0.5:  # 50% chance to record data every minute
                heart_rate_value = np.random.randint(60, 180)
                step_value = np.random.randint(0, 200)
                heart_rate_entries.append([person_id, current_date + timedelta(minutes=minute), heart_rate_value])
                step_entries.append([person_id, current_date + timedelta(minutes=minute), step_value])
        
        # Generate activity summary data once per day as before
        activity_summary_entries.append([
            person_id, 
            current_date.date(), 
            np.random.randint(1000, 2000),  # activity_calories
            np.random.randint(1000, 1500),  # calories_bmr
            np.random.randint(3000, 5000)   # steps
        ])
        
        # Increase the frequency and duration of sleep sessions
        num_sleep_sessions = np.random.randint(3, 7)  # 3 to 6 sleep sessions per day
        for _ in range(num_sleep_sessions):
            sleep_start_minute = np.random.randint(0, num_minutes)
            sleep_duration = np.random.randint(30, 360)  # Sleep duration between 30 to 360 minutes
            sleep_level = np.random.choice(['Awake', 'Light', 'REM', 'Deep'])
            sleep_entries.append([
                person_id,
                current_date,
                sleep_level,
                current_date + timedelta(minutes=sleep_start_minute),
                sleep_duration
            ])

# Convert lists to DataFrames
heart_rate_data = pd.DataFrame(heart_rate_entries, columns=['person_id', 'datetime', 'heart_rate_value'])
heart_rate_data['date'] = heart_rate_data['datetime'].dt.date
heart_rate_data['minute'] = heart_rate_data['datetime'].dt.hour * 60 + heart_rate_data['datetime'].dt.minute

step_data = pd.DataFrame(step_entries, columns=['person_id', 'datetime', 'step_value'])
step_data['date'] = step_data['datetime'].dt.date
step_data['minute'] = step_data['datetime'].dt.hour * 60 + step_data['datetime'].dt.minute

activity_summary_data = pd.DataFrame(activity_summary_entries, columns=['person_id', 'date', 'activity_calories', 'calories_bmr', 'steps'])

sleep_data = pd.DataFrame(sleep_entries, columns=['person_id', 'sleep_date', 'level', 'start_datetime', 'duration_in_min'])
sleep_data['sleep_date'] = pd.to_datetime(sleep_data['sleep_date'])
sleep_data['start_datetime'] = pd.to_datetime(sleep_data['start_datetime'])

samples_data = pd.DataFrame({
    'person_id': np.random.choice(np.arange(1, num_persons + 1), num_samples),
    'condition_start_datetime': pd.to_datetime('2021-01-01') + pd.to_timedelta(np.random.randint(1, num_days + 1, num_samples), unit='D')
})

# Map sleep levels to tensor dimensions
sleep_stage_to_dimension = {
    'Awake': 4,
    'Light': 5,
    'REM': 6,
    'Deep': 7
}

# Function to initialize tensors
def init_tensors(num_samples):
    return [torch.zeros((num_samples, num_minutes, d_model)) for _ in range(num_days)]

# Function to process a single sample
def process_sample(sample_index, row, tensors):
    person_id = row['person_id']
    endstamp = row['condition_start_datetime']
    start_stamp = endstamp - pd.Timedelta(days=num_days)

    person_heart_rate_data = heart_rate_data[(heart_rate_data['person_id'] == person_id) & 
                                             (heart_rate_data['datetime'] >= start_stamp) & 
                                             (heart_rate_data['datetime'] < endstamp)].copy()

    person_step_data = step_data[(step_data['person_id'] == person_id) & 
                                 (step_data['datetime'] >= start_stamp) & 
                                 (step_data['datetime'] < endstamp)].copy()

    person_activity_summary = activity_summary_data[(activity_summary_data['person_id'] == person_id) & 
                                                    (activity_summary_data['date'] >= start_stamp.date()) & 
                                                    (activity_summary_data['date'] < endstamp.date())].copy()

    person_heart_rate_data['day_index'] = (person_heart_rate_data['date'] - start_stamp.date()).apply(lambda x: x.days)
    person_step_data['day_index'] = (person_step_data['date'] - start_stamp.date()).apply(lambda x: x.days)
    
    person_sleep_data = sleep_data[sleep_data['person_id'] == person_id]

    for day_index in range(num_days):
        hr_day_mask = person_heart_rate_data['day_index'] == day_index
        hr_minute_indices = person_heart_rate_data.loc[hr_day_mask, 'minute'].values
        heart_rate_values = person_heart_rate_data.loc[hr_day_mask, 'heart_rate_value'].values

        step_day_mask = person_step_data['day_index'] == day_index
        step_minute_indices = person_step_data.loc[step_day_mask, 'minute'].values
        step_values = person_step_data.loc[step_day_mask, 'step_value'].values

        if hr_minute_indices.size > 0:
            tensors[day_index][sample_index, hr_minute_indices, 0] = torch.tensor(heart_rate_values, dtype=torch.float)

        if step_minute_indices.size > 0:
            tensors[day_index][sample_index, step_minute_indices, 1] = torch.tensor(step_values, dtype=torch.float)

        # Handle activity summary data
        summary_row = person_activity_summary[person_activity_summary['date'] == (start_stamp + pd.Timedelta(days=day_index)).date()]
        if not summary_row.empty:
            summary_row = summary_row.iloc[0]
            calories_bmr_per_minute = summary_row['calories_bmr'] / num_minutes

            day_step_data = person_step_data[person_step_data['day_index'] == day_index]
            day_heart_rate_data = person_heart_rate_data[person_heart_rate_data['day_index'] == day_index]

            merged_data = pd.merge(day_step_data, day_heart_rate_data, on='minute', how='outer', suffixes=('_step', '_hr'))
            merged_data = merged_data.sort_values('minute').reset_index(drop=True)

            active_minutes_mask = (merged_data['step_value'] > 20) | (merged_data['heart_rate_value'] > 100)
            active_minutes_indices = merged_data.loc[active_minutes_mask, 'minute'].values

            active_minutes = active_minutes_mask.sum()

            if active_minutes > 0:
                activity_calories_per_active_minute = summary_row['activity_calories'] / active_minutes
            else:
                activity_calories_per_active_minute = 0

            tensors[day_index][sample_index, :, 2] = calories_bmr_per_minute
            tensors[day_index][sample_index, active_minutes_indices, 3] = activity_calories_per_active_minute

        # Handle sleep data
        current_date = start_stamp + pd.Timedelta(days=day_index)
        daily_sleep_data = person_sleep_data[person_sleep_data['sleep_date'] == current_date]

        if not daily_sleep_data.empty:
            start_minutes = ((daily_sleep_data['start_datetime'] - daily_sleep_data['start_datetime'].dt.normalize()).dt.total_seconds() // 60).astype(int)
            end_minutes = start_minutes + daily_sleep_data['duration_in_min']
            sleep_levels = daily_sleep_data['level'].map(lambda x: sleep_stage_to_dimension[x]).values
            
            for start_minute, end_minute, sleep_level in zip(start_minutes, end_minutes, sleep_levels):
                if end_minute > num_minutes:
                    end_minute = num_minutes
                tensors[day_index][sample_index, start_minute:end_minute, sleep_level] = 1

# Function to count non-zero values in each dimension
def count_non_zero_elements(tensors):
    total_non_zero_count = {i: 0 for i in range(d_model)}
    for tensor in tensors:
        for dim in range(d_model):
            non_zero_count = (tensor[:, :, dim] != 0).sum().item()
            total_non_zero_count[dim] += non_zero_count
    return total_non_zero_count

# Function to calculate the average number of non-zero minutes per sample
def average_non_zero_minutes_per_sample(tensors, num_samples):
    total_non_zero_minutes = 0
    total_samples = num_samples * len(tensors)
    
    for tensor in tensors:
        # Count non-zero minutes across all dimensions for each sample
        non_zero_count = (tensor != 0).any(dim=2).sum().item()
        total_non_zero_minutes += non_zero_count

    avg_non_zero_minutes_per_sample = total_non_zero_minutes / total_samples
    return avg_non_zero_minutes_per_sample

if __name__ == "__main__":
    tensors = init_tensors(num_samples)

    for sample_index, row in samples_data.iterrows():
        process_sample(sample_index, row, tensors)

    non_zero_elements = count_non_zero_elements(tensors)
    print(f"Total number of non-zero elements in each dimension: {non_zero_elements}")

    avg_non_zero_minutes = average_non_zero_minutes_per_sample(tensors, num_samples)
    print(f"Average number of non-zero minutes per sample: {avg_non_zero_minutes}")

    print("Completed processing of all samples.")
