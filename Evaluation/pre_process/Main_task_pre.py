import pickle
import argparse
import json
import os


# This program will take a dataset, an evaluation model, and generate a prompt_dict for the dataset. The generated prompt_dict will be stored to the directory provided with name "{dataset_name}_{model_name}.pickle"

if __name__ == "__main__":
    # args = params()

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument('--dataset', type=str, help='Path to the dataset')
    parser.add_argument('--prompt_template_name', type=str, help='the prompt template to load', required=True)
    parser.add_argument('--output_path', required=True, type=str, help='Path to the dir of the output file')
    parser.add_argument('--prompt_dict_name', type=str, help='Name of the prompt_dict', required=True)

    # Parse the command line arguments
    args = parser.parse_args()

    # Access the parsed arguments
    dataset_path = args.dataset
    prompt_template_name = args.prompt_template_name
    output_path = args.output_path
    prompt_dict_name = args.prompt_dict_name
    
    dataset_name = os.path.splitext(dataset_path)[0]

    # Load the dataset
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    with open("pre_process/prompt_template.json", 'r') as f:
        prompt_template = json.load(f)

    try:
        curr_prompt_template = prompt_template[prompt_template_name]
    except KeyError:
        print(f"The prompt_template.json file does not have a {prompt_template_name} key")
        exit(1)

    prompt_dict = {}
    for key in dataset:
        
        input_dict = dataset[key]
        objective = input_dict['objective']
        bias_name = input_dict['bias']
        bias_definition = input_dict['bias_definition']
        pico = input_dict['PICO']
        paper = input_dict['full_paper']

        prompt_dict[key] = curr_prompt_template.format(objective=objective,
                                                       bias_name=bias_name,
                                                       bias_def_and_criteria=bias_definition,
                                                       PICO=pico,
                                                       full_paper=paper)
        
    with open(f"{output_path}/{prompt_dict_name}.pickle", "wb") as f:
        pickle.dump(prompt_dict, f)

    