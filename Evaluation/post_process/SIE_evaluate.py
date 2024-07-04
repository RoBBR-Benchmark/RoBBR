import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
from postprocess_util import get_west_coast_time, append_row_to_csv, bootstrap_mean_std_error, bootstrap_weighted_accuracy, bootstrap_f1_stats_macro,  bootstrap_accuracy
import pdb
import json

def f1_stats_macro(x, y):
    true_labels = y
    predicted_labels = x
    precision = precision_score(true_labels, predicted_labels, average="macro")
    recall = recall_score(true_labels, predicted_labels, average="macro")
    f1 = f1_score(true_labels, predicted_labels, average="macro")
    return f1


if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--input_path', default=None, type=str, required=True, help='the path to the parsed generation output')
    parser.add_argument('--dataset_path', default=None, type=str, required=True, help='the path to the dataset')
    parser.add_argument('--exp_name', type=str, required=True, help='the experiment name')
    parser.add_argument('--logging_csv', type=str, required=True, help='the path to the logging csv file')

    args = parser.parse_args()
    model_name = args.model_name
    input_path = args.input_path
    dataset_path = args.dataset_path
    exp_name = args.exp_name
    logging_csv_path = args.logging_csv
    
    with open(input_path, "rb") as f:
        parsed_list_dict = pickle.load(f)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    check_list = []
    model_output = []
    correct_output = []

    groundtruth_answer_dict = {'0':[], '1': []}

    for idx, key in enumerate(dataset):

        if parsed_list_dict[key] is None:
            # check_list.append(0)
            model_output.append(int(str(dataset[key]['label'] == "included").lower() != "true"))
            correct_output.append(int(str(dataset[key]['label'] == "included").lower() == "true"))
            continue

        parse_output = parsed_list_dict[key].lower()

        
        
        if 'true' not in parse_output and 'false' not in parse_output:
            print("No parsed output")
            print(f"Key: {key}")
            print(parse_output)
            print()

        curr_model_output = int(parse_output.lower() == "true")
        answer = int(dataset[key]['label'] == "included")

        model_output.append(curr_model_output)
        correct_output.append(answer)

        assert len(model_output) == len(correct_output)

    w_accuracy_mean, w_accuracy_ste = bootstrap_weighted_accuracy(model_output, correct_output)
    accuracy_mean, accuracy_ste = bootstrap_accuracy(model_output, correct_output)
    f1_mean, f1_ste = bootstrap_f1_stats_macro(model_output, correct_output)


    # print(f"Weighted accuracy: {w_accuracy_mean}+/-{w_accuracy_ste}")
    # print(f"Accuracy: {accuracy_mean}+/-{accuracy_ste}")
    # print(f"F1: {f1_mean}+/-{f1_ste}")
    
    result_dict = {"Experiment Name": exp_name ,"Model": model_name, "Time": get_west_coast_time()}
    
    result_dict['full_weighted_accuracy_mean'] = np.round(w_accuracy_mean*100, 2)
    result_dict['full_weighted_accuracy_ste'] = np.round(w_accuracy_ste*100, 2)
    result_dict['full_accuracy_mean'] = np.round(accuracy_mean*100, 2)
    result_dict['full_accuracy_ste'] = np.round(accuracy_ste*100, 2)
    result_dict['full_f1_mean'] = np.round(f1_mean*100, 2)
    result_dict['full_f1_ste'] = np.round(f1_ste*100, 2)

    for k in result_dict:
        print(f"{k}: {result_dict[k]}")


    append_row_to_csv(logging_csv_path, result_dict)

    



