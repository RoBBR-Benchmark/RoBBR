# Measuring Risk of Bias in Biomedical Reports: The RoBBR Benchmark

## Paper abstract
When expert reviewers assess the quality of evidence from biomedical reports, the risk-of-bias guideline is frequently used to determine if evidence from a report is biased. In our work, this decision-making process is reverse engineered to create RoBBR, a comprehensive risk-of-bias benchmark with three tasks that simulate human experts' workflow, created by a novel technique, GPT-Tracer. We evaluate a range of embedding and large language models (LLMs) on RoBBR, and reveal a large room for improvement. RoBBR can also be used to fine-tune language models.

## Dataset

All datasets are available on Huggingface at [RoBBR](https://huggingface.co/datasets/RoBBR-Benchmark/RoBBR). After downloading, place the datasets into the `dataset` folder.

###  Main Task: Risk-of-Bias Determination

The dataset structure is as follows:
- `paper_doi`: The DOI of the paper.
- `bias`: The bias to be considered.
- `bias_definition`: The risk of bias definition from the Cochrane Handbook.
- `PICO`: PICO of a study in the paper, including Methods, Participants, Intervention, Outcome, and Notes.
- `objective`: The meta-analysis objective.
- `full_paper`: The full paper content.
- `label`: One of [`low`, `high`, `unclear`], representing the risk level of the bias.

### Support Sentence Retrieval (SSR)

The dataset structure is as follows:
- `paper_doi`: The DOI of the paper.
- `bias`: The bias to be considered.
- `bias_definition`: The risk of bias definition from the Cochrane Handbook.
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

### Support Judgment Selection (SJS)

The dataset structure is as follows:
- `paper_doi`: The DOI of the paper.
- `bias`: The bias to be considered.
- `bias_definition`: The risk of bias definition from the Cochrane Handbook.
- `PICO`: PICO of a study in the paper, including Methods, Participants, Intervention, Outcome, and Notes.
- `objective`: The meta-analysis objective.
- `full_paper`: The full paper content.
- `options`: The seven options for the multiple choice.
- `label`: The index of the correct option.

## Evaluation

We provide end-to-end evaluation code for the tasks in our benchmark. The current pipelines for each of the tasks support the generation models mentioned in the paper. 

To run the evaluation for generation models, use the following command for each tasks:

### Main Task: Risk-of-Bias Determination
```bash
cd Evaluation
bash Main_task_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
All the arguments are same as the Task 1 case.

The results of all tasks will be recorded in `Evaluation/post_process/logs.csv` under the specified `exp_name`.

### Support Sentence Retrieval (SSR)
```bash
cd Evaluation
bash SSR_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name> <limits> <regeneration>
```
Most of the arguments are same as the Task 1 case.
`<limits>`: The limit used for calculating recall@limits during evaluation.

`<regeneration>` A boolean indicating whether the model will regenerate if it retrieves more than the specified number of sentences.

### Support Judgment Selection (SJS)
```bash
cd Evaluation
bash SJS_eval.sh <dataset_path> <max_tokens> <prompt_template_name> <model_name> <exp_name>
```
All the arguments are same as the Task 1 case.

## Model Checkpoints

Our finetuned model checkpoints are available at [huggingface](https://huggingface.co/RoBBR-Benchmark).

We provide three checkpoints [RoBBR-Benchmark/llama3-8B_main_task](https://huggingface.co/RoBBR-Benchmark/llama3-8B_main_task), [RoBBR-Benchmark/llama3-8B_ssr_task](https://huggingface.co/RoBBR-Benchmark/llama3-8B_ssr_task), [RoBBR-Benchmark/llama3-8B_sjs_task](https://huggingface.co/RoBBR-Benchmark/llama3-8B_sjs_task), that are finetuned on the three tasks.

## Model Fine-tuning

We used torchtune to fine-tune Llama-3-8B with LoRA. Firstly, install PyTorch 2.4.0 and torchtune 0.2.1.

After installation, download the pretrained model by running the following command:
```
tune download meta-llama/Meta-Llama-3-8B-Instruct --output-dir /tmp/Meta-Llama-3-8B-Instruct --hf-token <HF_TOKEN>
```
Note, the download requires access to `meta-llama/Meta-Llama-3-8B-Instruct` on Hugging Face.

After changing directories for training data in `fine_tuning/train_data.py` and checkpoint/logging configurations in `fine-tuning/tune_llama_8B_lora.yaml`, run the following commnad to fine-tune the model:
```
tune run --nproc_per_node <number of GPUs> lora_finetune_distributed --config fine_tuning/tune_llama_8B_lora.yaml
```