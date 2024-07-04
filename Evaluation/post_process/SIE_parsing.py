import pickle
import json
import numpy as np
import re
from sympy import limit
from tqdm import tqdm
import argparse
import os
from postprocess_util import parse_task_1_output
from best_effort_string_match import find_matched_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--model_output_path', default=None, type=str, required=True, help='the path to read the generation output')
    parser.add_argument('--parsed_output_path', default=None, type=str, required=True, help='the path to store the parsed output')    
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset')

    args = parser.parse_args()
    model_name = args.model_name
    model_output_path = args.model_output_path
    parsed_output_path = args.parsed_output_path
    dataset_path = args.dataset_path

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    with open(model_output_path, "rb") as f:
        model_output = pickle.load(f)

    parsed_list_dict = {}
    for prompt_id in model_output:
        parsed_output = parse_task_1_output(model_output[prompt_id])
        parsed_list_dict[prompt_id] = parsed_output

    with open(parsed_output_path, "wb") as f:
        pickle.dump(parsed_list_dict, f)

