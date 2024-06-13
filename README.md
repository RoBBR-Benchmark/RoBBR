# Measuring Risk of Bias in Biomedical Reports: The RoBBR Benchmark

## Paper abstract
Systems that answer questions by reviewing the scientific literature are becoming increasingly feasible. To draw reliable conclusions, these systems should take into account the quality of available evidence, placing more weight on studies that use a valid methodology. We present a benchmark for measuring the methodological strength of biomedical papers, drawing on the risk-of-bias framework used for systematic reviews. The four benchmark tasks, drawn from more than 500 papers, cover the analysis of research study methodology, followed by evaluation of risk of bias in these studies. The benchmark contains 2000 expert-generated bias annotations, and a human-validated pipeline for fine-grained alignment with research paper content. We evaluate a range of large language models on the benchmark, and find that these models fall significantly short of human-level performance. By providing a standardized tool for measuring judgments of study quality, the benchmark can help to guide systems that perform large-scale aggregation of scientific data.


## Evaluation

We provide end-to-end evaluation code for the tasks in our benchmark. The current pipelines for each of the tasks support the generation models mentioned in the paper. 

To run the evaluation for generation models, use the following command for each tasks:

### Task 1: SIE
```bash
cd Evaluation
bash task_1_SIE_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
`<dataset_path>`: The path to the dataset to evaluate, typically located in the `dataset/` directory.

`<max_tokens>`: The maximum tokens for the generation models.

`<prompt_template_name>` The prompt template to use for generation.

`<model_name>`: The name of the generation model used for evaluation. Supported models include: `['gpt-4o-2024-05-13', 'claude-3-opus-20240429', 'gemini-1.5-pro-latest']`

`<exp_name>`: The name under which the evaluation results will be documented in the experiment logs.

### Task 2: ROBSR
```bash
cd Evaluation
bash task_2_ROBSR_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name> <limits> <regeneration>
```
Most of the arguments are same as the Task 1 case.
`<limits>`: The limit used for calculating recall@limits during evaluation.

`<regeneration>` A boolean indicating whether the model will regenerate if it retrieves more than the specified number of sentences.

### Task 3: SJS
```bash
cd Evaluation
bash task_3_SJS_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
All the arguments are same as the Task 1 case.

### Task 4: RLD
```bash
cd Evaluation
bash task_4_RLD_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
All the arguments are same as the Task 1 case.

The results of both all tasks will be recorded in `Evaluation/post_process/logs.csv` under the specified `exp_name`.
