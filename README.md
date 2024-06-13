# Measuring Risk of Bias in Biomedical Reports: The RoBBR Benchmark

## Paper abstract
Systems that answer questions by reviewing the scientific literature are becoming increasingly feasible. To draw reliable conclusions, these systems should take into account the quality of available evidence, placing more weight on studies that use a valid methodology. We present a benchmark for measuring the methodological strength of biomedical papers, drawing on the risk-of-bias framework used for systematic reviews. The four benchmark tasks, drawn from more than 500 papers, cover the analysis of research study methodology, followed by evaluation of risk of bias in these studies. The benchmark contains 2000 expert-generated bias annotations, and a human-validated pipeline for fine-grained alignment with research paper content. We evaluate a range of large language models on the benchmark, and find that these models fall significantly short of human-level performance. By providing a standardized tool for measuring judgments of study quality, the benchmark can help to guide systems that perform large-scale aggregation of scientific data.

## Dataset Structure

### Task 1: Study Inclusion/Exclusion (SIE)
The dataset is available at `dataset/task1_SIE_test.json` and `dataset/task1_SIE_dev.json`
Test and development set for Task 1:
- `paper_doi`: The DOI of the paper.
- `objective`: The meta-analysis objective.
- `search_protocol`: Search protocol information of the meta-analysis.
- `full_paper`: The full paper content.
- `label`: One of [`included`, `excluded`], showing whether the paper is included or excluded in the meta-analysis.

### Task 2: Risk of Bias Sentence Retrieval (ROBSR)
The dataset is available at `dataset/task2_ROBSR_test.json` and `dataset/task2_ROBSR_dev.json`
Test and development set for Task 2:
- `paper_doi`: The DOI of the paper.
- `bias`: The bias to be considered.
- `PICO`: PICO of a study in the paper, including Methods, Participants, Intervention, Outcome, and Notes.
- `objective`: The meta-analysis objective.
- `paper_as_candidate_pool`: A tuple of text elements from the paper. Each text element is a sentence, a section title, a table, or a figure caption.
- `aspects`: A dictionary that maps aspect id to bias aspect.
- `aspect2sentence_indices`: A mapping between aspect id and all sentence indices that independently are a source of information for that aspect, as annotated by our pipeline.
- `sentence_index2aspects`: A mapping between sentence index and all aspect ids that this sentence is the source of information of.
- `bias_retrieval_at_optimal_evaluation`: A dictionary containing the necessary information for evaluating the model's performance on the task Bias Retrieval @Optimal.
  - `optimal`: A positive integer, which is the smallest number of sentences needed to cover the largest number of aspects.
  - `one_selection_of_sentences`: A list of sentence indices. The list size is the optimal number. The list of sentences cover the largest number of aspects.
  - `covered_aspects`: The list of aspects that are covered.
- `bias_retrieval_at_3_evaluation`: A dictionary containing the necessary information for evaluating your model's performance on the task Bias Retrieval @3.
  - `one_selection_of_sentences`: A list of 3 sentence indices. The list of sentences cover the largest number of aspects that can be covered under the restriction of 3 sentences.
  - `covered_aspects`: The list of aspects that are covered.

### Task 3: Support Judgment Selection (SJS)
The dataset is available at `dataset/task3_SJS_test.json` and `dataset/task3_SJS_dev.json`
Test and development set for Task 3:
- `paper_doi`: The DOI of the paper.
- `bias`: The bias to be considered.
- `PICO`: PICO of a study in the paper, including Methods, Participants, Intervention, Outcome, and Notes.
- `objective`: The meta-analysis objective.
- `full_paper`: The full paper content.
- `options`: The seven options for the multiple choice.
- `label`: The index of the correct option.

### Task 4: Risk Level Determination (RLD)
The dataset is available at `dataset/task4_RLD_test.json` and `dataset/task4_RLD_dev.json`
Test and development set for Task 4:
- `paper_doi`: The DOI of the paper.
- `bias`: The bias to be considered.
- `PICO`: PICO of a study in the paper, including Methods, Participants, Intervention, Outcome, and Notes.
- `objective`: The meta-analysis objective.
- `full_paper`: The full paper content.
- `label`: One of [`low`, `high`, `unclear`], representing the risk level of the bias.

### The Subsets of the Test Set Used in Our Paper
Due to budget limits, we evaluate the models using only a subset of the test set in our paper. 
We provide this subset as a list of keys at `dataset/task1_subset_used_in_main_paper.json`, `dataset/task2_subset_used_in_main_paper.json`, `dataset/task3_subset_used_in_main_paper.json`, and `dataset/task4_subset_used_in_main_paper.json`.

## Evaluation

We provide end-to-end evaluation code for the tasks in our benchmark. The current pipelines for each of the tasks support the generation models mentioned in the paper. 

To run the evaluation for generation models, use the following command for each tasks:

### Task 1: Study Inclusion/Exclusion (SIE)
```bash
cd Evaluation
bash task_1_SIE_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
`<dataset_path>`: The path to the dataset to evaluate, typically located in the `dataset/` directory.

`<max_tokens>`: The maximum tokens for the generation models.

`<prompt_template_name>` The prompt template to use for generation.

`<model_name>`: The name of the generation model used for evaluation. Supported models include: `['gpt-4o-2024-05-13', 'claude-3-opus-20240429', 'gemini-1.5-pro-latest']`

`<exp_name>`: The name under which the evaluation results will be documented in the experiment logs.

### Task 2: Risk of Bias Sentence Retrieval (ROBSR)
```bash
cd Evaluation
bash task_2_ROBSR_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name> <limits> <regeneration>
```
Most of the arguments are same as the Task 1 case.
`<limits>`: The limit used for calculating recall@limits during evaluation.

`<regeneration>` A boolean indicating whether the model will regenerate if it retrieves more than the specified number of sentences.

### Task 3: Support Judgment Selection (SJS)
```bash
cd Evaluation
bash task_3_SJS_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
All the arguments are same as the Task 1 case.

### Task 4: Risk Level Determination (RLD)
```bash
cd Evaluation
bash task_4_RLD_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
All the arguments are same as the Task 1 case.

The results of all tasks will be recorded in `Evaluation/post_process/logs.csv` under the specified `exp_name`.
