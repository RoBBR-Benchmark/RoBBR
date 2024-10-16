import numpy as np
import os
from datetime import datetime
import pytz
import csv
import pickle
import re

import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils import resample

def bootstrap_mean_std_error(data, num_samples=1000):
    """
    Perform bootstrapping to find the mean and standard error of the given data.
    
    Parameters:
    data (list or numpy array): The data points to bootstrap.
    num_samples (int): The number of bootstrap samples to generate (default is 1000).
    
    Returns:
    tuple: The mean and standard error of the bootstrap samples.
    """
    
    bootstrapped_means = []
    
    # Perform bootstrapping
    for _ in range(num_samples):
        # Generate a bootstrap sample by sampling with replacement from the data
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the bootstrap sample
        bootstrapped_mean = np.mean(bootstrap_sample)
        bootstrapped_means.append(bootstrapped_mean)
    
    # Calculate the mean and standard error of the bootstrap means
    mean_of_means = np.mean(bootstrapped_means)
    std_error_of_means = np.std(bootstrapped_means)
    
    return mean_of_means, std_error_of_means

def bootstrap_accuracy(model_result, gt_result, n_iterations=1000):
    """
    Calculate the accuracy of the model predictions based on the ground truth labels.
    Compute the standard error of the accuracy using bootstrapping.
    """
    assert len(model_result) == len(gt_result)
    accuracies = []

    for _ in range(n_iterations):
        # Bootstrap sample
        resampled_idx = resample(range(len(model_result)), replace=True, n_samples=len(model_result))
        resampled_model_result = [model_result[i] for i in resampled_idx]
        resampled_gt_result = [gt_result[i] for i in resampled_idx]

        l = [int(resampled_model_result[idx] == resampled_gt_result[idx]) for idx in range(len(resampled_model_result))]
        accuracies.append(np.mean(l))

    l = [int(model_result[idx] == gt_result[idx]) for idx in range(len(model_result))]

    return np.mean(l), np.std(accuracies)

def bootstrap_f1_stats_macro(x, y, n_iterations=1000):
    """
    Calculate the F1 score of the model predictions based on the ground truth labels.
    Compute the standard error of the F1 score using bootstrapping.
    """
    f1_scores = []

    for _ in range(n_iterations):
        # Bootstrap sample
        resampled_idx = resample(range(len(x)), replace=True, n_samples=len(x))
        resampled_x = [x[i] for i in resampled_idx]
        resampled_y = [y[i] for i in resampled_idx]

        f1 = f1_score(resampled_y, resampled_x, average="macro")
        f1_scores.append(f1)

    return f1_score(y, x, average="macro"), np.std(f1_scores)

def parse_main_output(text):
    text = text.replace("*", "")
    pattern = r'DECISION:\s*(.*?)(?:\s*\n|$)'
    match = re.search(pattern, text)
    if match:
        decision_text = match.group(1)
        return decision_text.lower()
    return None

def parse_SJS_output(text):
    text = text.replace("*", "")
    pattern = r'DECISION:\s*(.*?)(?:\s*\n|$)'
    match = re.search(pattern, text)
    if match:
        decision_text = match.group(1)
        if len(decision_text) > 1:
            return "H"
        return decision_text
    return "H"

def parse_claude_output(prompt, text):
    # This regex looks for the word 'DECISION' followed by a colon and brackets containing numbers separated by commas
    pattern = r'DECISION:.*?\[([0-9, ]+)\]'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        numbers = match.group(1).split(',')
        numbers = [int(num.strip()) for num in numbers]
        return numbers
    return []

def parse_gpt_output(prompt, text):
    # This regex looks for the word 'DECISION' followed by a colon and brackets containing numbers separated by commas
    text = text.replace("```python", "")
    text = text.replace("```", "")
    text = text.replace("*", "")
    text = text.replace("index", "")
    pattern = r'DECISION:\s*[-*]*\s*\[([0-9, ]+)\]'

    # Find all matches of the pattern
    matches = re.findall(pattern, text)

    # Check if there are any matches and extract numbers from the last match
    if matches:
        # Get the last match
        last_match = matches[-1]
        numbers = [int(num.strip()) for num in last_match.split(',')]
        return numbers
    return []

def parse_gemini_output(prompt, text):
    text = text.replace("Revised Selection", "DECISION").replace("Decision", "DECISION")
    pattern = r'(?:\*\*|##)?\s*DECISION:\s*[-*]*\s*\[([0-9, ]+)\]'

    # Find all matches of the pattern
    matches = re.findall(pattern, text)

    # Check if there are any matches and extract numbers from the last match
    if matches:
        # Get the last match
        
        last_match = matches[-1]

        if last_match.strip() == "":
            return []
        # Split the last match into individual items (numbers or ranges)
        items = last_match.split(',')
        numbers = []

        for item in items:
            item = item.strip()
            if '-' in item:
                # Expand the range
                start, end = map(int, item.split('-'))
                numbers.extend(range(start, end + 1))
            else:
                # Convert single number to integer and add to the list
                numbers.append(int(item))
        return numbers
    return []

def get_west_coast_time():
    # Define the timezone for US West Coast (Pacific Time)
    pacific_time = pytz.timezone('America/Los_Angeles')

    # Get the current time in Pacific Time
    now = datetime.now(pacific_time)

    # Format the time as mm_dd_hour_minute
    formatted_time = now.strftime('%m_%d_%H_%M')
    
    return formatted_time

def append_row_to_csv(file_path, row_dict):
    # Read the existing CSV file

    file_exists = os.path.exists(file_path)

    if not file_exists:
        column_names = ['Experiment Name', 'Time']

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(column_names)
        print(f'CSV file {file_path} created with column name: {column_names}')
    
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames

        # Update the fieldnames with new columns from the dictionary
        new_fieldnames = set(row_dict.keys()) - set(fieldnames)
        fieldnames.extend(new_fieldnames)

        rows = list(reader)

    # Create a new dictionary for the row with missing keys set to empty strings
    new_row = {key: row_dict.get(key, '') for key in fieldnames}

    # Append the new row to the existing rows
    rows.append(new_row)

    # Write the updated rows back to the CSV file
    with open(file_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_multi_turn_prompt_to_fix_output(prompt_key, next_round_prompt, init_prompt_path, model_output_path):
    with open(init_prompt_path, 'rb') as f:
        init_prompt_dict = pickle.load(f)
    with open(model_output_path, 'rb') as f:
        output_dict = pickle.load(f)

    init_prompt = init_prompt_dict[prompt_key]
    output = output_dict[prompt_key]

    new_prompt = {}

    new_prompt['conversation'] = []

    new_prompt['conversation'].append({'role': 'user', 'content': init_prompt})
    new_prompt['conversation'].append({'role': 'assistant', 'content': output})
    new_prompt['conversation'].append({'role': 'user', 'content': next_round_prompt})

    return new_prompt
