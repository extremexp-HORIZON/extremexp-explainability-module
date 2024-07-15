import pandas as pd
import json
from datetime import datetime, timedelta
import random

# Read the CSV file
csv_file_path = 'I2Cat_metrics.csv'
data = pd.read_csv(csv_file_path)

# Function to generate a random timestamp
def random_timestamp():
    return datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 24))

# Calculate start and end times based on runtime
def calculate_timestamps(runtime):
    start_time = random_timestamp()
    end_time = start_time + timedelta(seconds=runtime)
    return start_time, end_time

# Shuffle rows for randomness
data_shuffled = data.sample(frac=1).reset_index(drop=True)

# Initialize status counts
running_count = 0
failed_count = 0
completed_count = 0

# Create the JSON structure
json_data = []

for index, row in data_shuffled.iterrows():
    status = "completed"
    workflow_info = {
        "startTime": random_timestamp().isoformat(),
        "status": None,
        "endTime": None,
        "completedTasks": None,
        "scheduledPosition": None
    }
    
    constraints = {
        "runtimeLess1sec": row['runtime'] < 1
    }
    
    metrics = None
    
    if running_count < 2 and status == "completed":
        status = "running"
        running_count += 1
        workflow_info["completedTasks"] = "2/4"
    elif failed_count < 2 and status == "completed":
        status = "failed"
        failed_count += 1
        start_time, end_time = calculate_timestamps(row['runtime'])
        workflow_info["startTime"] = start_time.isoformat()
        workflow_info["endTime"] = end_time.isoformat()
    elif status == "completed":
        completed_count += 1
        start_time, end_time = calculate_timestamps(row['runtime'])
        workflow_info["startTime"] = start_time.isoformat()
        workflow_info["endTime"] = end_time.isoformat()
        metrics = {
            "accuracy": row['accuracy'],
            "precision": row['precision'],
            "recall": row['recall'],
            "f1_score": row['f1_score'],
            "f1_macro": row['f1_macro'],
            "true_positives": row['true_positives'],
            "false_positives": row['false_positives'],
            "true_negatives": row['true_negatives'],
            "false_negatives": row['false_negatives'],
            "runtime": row['runtime']
        }

    workflow_info["status"] = status
    
    entry = {
        "workflowId": row['id'],
        "variabilityPoints": {
            "max_depth": row['max_depth'],
            "min_child_weight": row['min_child_weight'],
            "learning_rate": row['learning_rate'],
            "n_estimators": row['n_estimators'],
            "scaler": row['scaler']
        },
        "metrics": metrics,
        "workflowInfo": workflow_info,
        "constraints": constraints
    }
    json_data.append(entry)

# Handle scheduled ones
scheduled_positions = list(range(5))
for i in range(1, 6):
    json_data[-i]["workflowInfo"]["status"] = "scheduled"
    json_data[-i]["workflowInfo"]["scheduledPosition"] = scheduled_positions.pop(0)
    json_data[-i]["workflowInfo"]["startTime"] = None
    json_data[-i]["workflowInfo"]["endTime"] = None
    json_data[-i]["workflowInfo"]["completedTasks"] = None
    json_data[-i]["metrics"] = None  # Ensure metrics is not populated for scheduled workflows

# Convert the list to JSON
json_output = json.dumps(json_data, indent=4)

# Write the JSON to a file
with open('workflows.json', 'w') as json_file:
    json_file.write(json_output)

print("JSON data has been written to workflows.json")
