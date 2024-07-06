#!/bin/bash
set -e

# Function to log the PID of the current script and subprocesses
log_pid() {
    echo "$1: $2" >> "$pid_file"
}

# Check if exactly all arguments are provided
if [ $# -ne 5 ]; then
    echo "Usage: $0 <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>"
    exit 1
fi

# Assign arguments to descriptive named variables
dataset_path="$1"
max_tokens="$2"
prompt_template_name="$3"
model_name="$4"
exp_name="$5"

# Get the current date and time in Eastern Time
current_time=$(TZ="America/Los_Angeles" date +"%Y-%m-%d_%H-%M-%S")
pid_file="logs/PIDs/PIDs_from_end_to_end_eval_${exp_name}_${current_time}.pids"

# Log the PID of the current script
log_pid "Main Script" $$
echo "========================== RLD =========================="

# Print the variable names and values
echo "Dataset path: $dataset_path"
echo "Max tokens: $max_tokens"
echo "Prompt template name: $prompt_template_name"
echo "Generation model name: $model_name"
echo "Experiment name: $exp_name"

echo "=================================================================="

# Create prompt dict for generation
python3 pre_process/RLD_pre.py --dataset "$dataset_path" --prompt_template_name "$prompt_template_name" --prompt_dict_name "$exp_name" --output_path "./generation/prompts/"

echo "Finish RLD prompt dict generation..."
echo "=================================================================="

# # LLM Generation
python3 generation/generation.py --prompt_path "generation/prompts/${exp_name}.pickle" --output_path "generation/outputs/${exp_name}" --max_workers 25 --timeout_seconds 240 --generation_model ${model_name} --temperature 0 --max_tokens ${max_tokens} & 
pid=$!
log_pid "generation/generation.py" $pid
wait $pid

echo "Finish generation..."
echo "=================================================================="

# Parsing
python3 post_process/RLD_parsing.py --model_name ${model_name} --model_output_path "generation/outputs/${exp_name}/collected_results.pickle" --parsed_output_path "./post_process/parsed_output/${exp_name}.pickle" --dataset_path ${dataset_path}

echo "Finish parsing..."
echo "=================================================================="

echo "Start evaluation..."

# # Post process and evaluation
python3 post_process/RLD_evaluate.py --model_name ${model_name} --input_path "./post_process/parsed_output/${exp_name}.pickle" --dataset_path ${dataset_path} --exp_name ${exp_name} --logging_csv "post_process/logs.csv"

echo "Finish post process and evaluation..."
echo "The results will be added to post_process/logs.csv"
echo "===============================Finished=================================="

# # Note: The script will stop executing at the first command that fails due to 'set -e'.

