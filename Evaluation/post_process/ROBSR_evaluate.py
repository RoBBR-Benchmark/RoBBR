import argparse
import pickle
import random
import numpy as np
from tqdm import tqdm
import warnings
from postprocess_util import get_west_coast_time, append_row_to_csv, bootstrap_mean_std_error, bootstrap_weighted_accuracy, bootstrap_f1_stats_macro,  bootstrap_accuracy
import pdb
import json

with open("post_process/bias_category_mapping.pickle", 'rb') as f:
    bias_category = pickle.load(f)

b2c = bias_category['bias2category']


import random
def calculate_per_category_result_ROBSR(dataset, parsed_output_path, b2c, limit=-1):
    random.seed(123)
    np.random.seed(123)
    with open(parsed_output_path, 'rb') as f:
        parsed_output = pickle.load(f)

    
    category_recall = {}

    full_recall = []

    for k,v in dataset.items():

        if limit==-1:
            curr_limit = v['bias_retrieval_at_optimal_evaluation']['optimal']
        else:
            curr_limit = limit

        curr_model_output = parsed_output[k]
        curr_bias_name = v['bias']
        curr_category = b2c[curr_bias_name]

        max_asp_to_cover = 0
        for aid in v['aspect2sentence_indices']:
            if len(v['aspect2sentence_indices'][aid]) > 0:
                max_asp_to_cover += 1

        if len(curr_model_output) > curr_limit:
            # this is the case model returns more than curr_limit k points
            curr_recall_list = []
            for _ in range(10000):
                random.shuffle(curr_model_output)
                temp_model_output = curr_model_output[:curr_limit]
                asp_model_can_cover = []
                for sid in temp_model_output:
                    asp_model_can_cover += v['sentence_index2aspects'][str(sid)]

                asp_model_can_cover = list(set(asp_model_can_cover))

                curr_recall_list.append(len(asp_model_can_cover)/max_asp_to_cover)
    
            curr_recall = np.mean(curr_recall_list)
        
        else:
            asp_model_can_cover = []
            for sid in curr_model_output:
                asp_model_can_cover += v['sentence_index2aspects'][str(sid)]

            asp_model_can_cover = list(set(asp_model_can_cover))

            curr_recall = len(asp_model_can_cover)/max_asp_to_cover

        full_recall.append(curr_recall)

        for cat in curr_category:
            if cat not in category_recall:
                category_recall[cat] = []

            category_recall[cat].append(curr_recall)

    result = {}

    for cat in category_recall:
        result[cat] = {}
        curr_recall = category_recall[cat]
        
        result[cat]['num_of_points'] = len(curr_recall)
        # recall_mean, recall_ste = bootstrap_mean_std_error(curr_recall)
        recall_mean = np.mean(curr_recall)
        recall_ste = np.std(curr_recall)/np.sqrt(len(curr_recall))
        result[cat]['recall_mean'] = recall_mean
        result[cat]['recall_ste'] = recall_ste

    print(full_recall)

    # full_recall_mean, full_recall_ste = bootstrap_mean_std_error(full_recall)
    full_recall_mean = np.mean(full_recall)
    full_recall_ste = np.std(full_recall)/np.sqrt(len(full_recall))
    result['full'] = {}
    result['full']['num_of_points'] = len(full_recall)
    result['full']['recall_mean'] = full_recall_mean
    result['full']['recall_ste'] = full_recall_ste

    return result




if __name__ == "__main__":
    random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default=None, type=str, required=True, help="the model used for generation")
    parser.add_argument('--input_path', default=None, type=str, required=True, help='the path to the parsed generation output')
    parser.add_argument('--dataset_path', default=None, type=str, required=True, help='the path to the dataset')
    parser.add_argument('--exp_name', type=str, required=True, help='the experiment name')
    parser.add_argument('--logging_csv', type=str, required=True, help='the path to the logging csv file')
    parser.add_argument('--limit_k', default=20, type=int, help='the number of sentences the model should output')
    # parser.add_argument('--output_path', default=None, type=str, required=True, help='the path to the parsed output')

    args = parser.parse_args()
    model_name = args.model_name
    input_path = args.input_path
    dataset_path = args.dataset_path
    exp_name = args.exp_name
    logging_csv_path = args.logging_csv
    sent_limit = args.limit_k

    with open(input_path, "rb") as f:
        parsed_list_dict = pickle.load(f)
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    
    ret = calculate_per_category_result_ROBSR(dataset, input_path, b2c, sent_limit)

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

    



