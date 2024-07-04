import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm
import warnings
from sklearn.metrics import precision_score, recall_score, f1_score
import json

from postprocess_util import get_west_coast_time, append_row_to_csv, bootstrap_mean_std_error, bootstrap_weighted_accuracy, bootstrap_f1_stats_macro,  bootstrap_accuracy
import pdb

with open("post_process/bias_category_mapping.pickle", 'rb') as f:
    bias_category = pickle.load(f)

b2c = bias_category['bias2category']


def calculate_per_category_result_SJS(dataset, parsed_output_path, b2c):
    random.seed(123)
    np.random.seed(123)
    with open(parsed_output_path, 'rb') as f:
        parsed_output = pickle.load(f)

    category_model_output = {}
    category_correct_answer = {}

    full_model_output = []
    full_correct_output = []

    model_answer2num = {'low': 0, 'unclear': 1, 'high': 2}

    for k,v in dataset.items():

        curr_bias_name = v['bias'].lower()
        curr_model_output = parsed_output[k]

        model_answer = curr_model_output
        gt_answer = v['label']

        

        curr_category = b2c[curr_bias_name]
        full_model_output.append(model_answer)
        full_correct_output.append(gt_answer)

        for cat in curr_category:
            if cat not in category_model_output:
                category_model_output[cat] = []
                category_correct_answer[cat] = []

            category_model_output[cat].append(model_answer)
            category_correct_answer[cat].append(gt_answer)

    result = {}
    

    for cat in category_model_output:
        result[cat] = {}
        curr_cat_model_output = category_model_output[cat]
        curr_cat_gt_answer = category_correct_answer[cat]
        result[cat]['num_of_points'] = len(curr_cat_model_output)


        acc_mean, acc_ste = bootstrap_accuracy(curr_cat_model_output, curr_cat_gt_answer, n_iterations=1000)

        result[cat]['accuracy_mean'] = acc_mean
        result[cat]['accuracy_ste'] = acc_ste



    acc_mean, acc_std = bootstrap_accuracy(full_model_output, full_correct_output)
    
    result['full'] = {}
    result['full']['num_of_points'] = len(full_model_output)
    result['full']['accuracy_mean'] = acc_mean
    result['full']['accuracy_ste'] = acc_std


    return result


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

    ret = calculate_per_category_result_SJS(dataset, input_path, b2c)

    result_dict = {"Experiment Name": exp_name ,"Model": model_name, "Time": get_west_coast_time()}
    for k in ret:
        for sub_k in ret[k]:
            if sub_k != 'num_of_points':
                result_dict[f"{k}_{sub_k}"] = np.round(ret[k][sub_k]*100, 2)
                print(f"{k}_{sub_k}: {np.round(ret[k][sub_k]*100, 2)}")
            else:
                result_dict[f"{k}_{sub_k}"] = ret[k][sub_k]
                print(f"{k}_{sub_k}: {ret[k][sub_k]}")

        print('-'*30)

    append_row_to_csv(logging_csv_path, result_dict)

    



