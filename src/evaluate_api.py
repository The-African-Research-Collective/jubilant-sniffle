import os
import json
import asyncio
import wandb
import logging
import pandas as pd

from datasets import load_dataset
from string import Template
from copy import copy
from args import load_config
from utils import generate_lang_task_list, MAFAND_CODE_2_LANG
from processing_queue import MessageQueue
from tqdm import tqdm
from collections import defaultdict
import evaluate


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

METRICS = [ 'acc', 'exact_match', 'f1', 'acc_stderr', 'ter', 'bleu', 'chrf']
TRANSLATION_METRICS = ['ter', 'bleu', 'chrf']

METRIC_MAPPING = {
    metric: evaluate.load_metric(metric) for metric in METRICS
}

def process_translation_example_prompt_mafand(example, language_pair, prompts):

    def format_prompt(prompt, source_sentence, source_language, target_language):
        return prompt.substitute(source_sentence=source_sentence, source_language=source_language, target_language=target_language)
    
    source_code = language_pair.split('_')[0]
    target_code = language_pair.split('_')[1]

    for prompt_file in prompts:
        source_sentence = example['translation'][source_code]
        target_sentence = example['translation'][target_code]
        source_language = MAFAND_CODE_2_LANG[source_code]
        target_language = MAFAND_CODE_2_LANG[target_code]

        example[f"{prompt_file}_{source_code}_{target_code}"] = format_prompt(copy(prompts[prompt_file]), source_sentence, source_language, target_language)
        example[f"{prompt_file}_{target_code}_{source_code}"] = format_prompt(copy(prompts[prompt_file]), target_sentence, target_language, source_language)
    
    return example



def generate_wandb_run_name(model_args: str, num_few_shot: int):
    model_args = model_args.replace('pretrained=', '')
    org_model_name = model_args.split(',')[0]

    model_name = org_model_name.split('/')[-1].replace('_', '-')

    return f"together_api-{model_name}-{num_few_shot}shot"

async def main():
    config = load_config()
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Batch size: {config.eval.batch_size}")
    logger.info(f"Random seed: {config.eval.random_seed}")
    logger.info(f"Use cache: {config.eval.use_cache}")
    logger.info(f"num_fewshot: {config.eval.num_fewshot}")

    task_list, _ = generate_lang_task_list(config.task)

    logger.info(f"Task list: {task_list}")

    if not config.eval.split_tasks:
        task_list = [task_list]
    
    metric_df_path = os.path.join(config.run_dir, "metrics_df.json")
    metric_results_path = os.path.join(config.run_dir, "metric_results.json")
    samples_path = os.path.join(config.run_dir, "samples.json")
    configs_path = os.path.join(config.run_dir, "configs.json")

    if os.path.exists(metric_df_path):
        metrics_df = pd.read_json(metric_df_path, lines=True)
        metric_results = json.load(open(metric_results_path, "r"))
        samples = json.load(open(samples_path, "r"))
        configs = json.load(open(configs_path, "r"))
    else:
        metric_results = {}
        metrics_df = None
        samples = {}
        configs = {}
    
    # load prompts from prompt dir
    prompt_dir = config.task.prompt_directory

    prompts = {}
    if prompt_dir:
        prompt_files_list = os.listdir(prompt_dir)
        for prompt_file in prompt_files_list:
            with open(os.path.join(prompt_dir, prompt_file), 'r') as f:
                prompts[prompt_file] = Template(f.read())
    else:
        raise ValueError("Prompt directory not provided")

    
    queue = MessageQueue(db_path=os.path.join(config.run_dir, f"{config.task.task_name}_{config.model.model_name.replace("/", "_")}.db")  , batch_size=32)

    def batch_iterable(iterable, n=32):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    processed_datasets = []
    for task in config.task.languages:
        task_dataset = load_dataset(config.task.hub_dataset, task.replace('_', '-'))
        task_dataset = task_dataset[config.task.evaluation_split]
        initial_columns = set(task_dataset.column_names)

        task_dataset = task_dataset.map(process_translation_example_prompt_mafand, fn_kwargs={'language_pair': task, 'prompts': prompts})
        prompt_columns = set(task_dataset.column_names) - set(initial_columns)

        processed_datasets.append((task_dataset, prompt_columns))

        for column in prompt_columns:
            for text_batch in tqdm(batch_iterable(task_dataset[column]), desc=f"Processing {column}"):
                translation_direction = "_".join(column.split('_')[-2:])
                prompt_type = "_".join(column.split('_')[:-2])
                await queue.enqueue(text_batch, translation_direction, prompt_type)

        stats = await queue.process_queue(model_name=config.model.model_name)

        logger.info(f"Run Stats: {stats}")
    
    
    # Evaluate the model
    samples = defaultdict(list)
    
    for dataset, prompt_columns in processed_datasets:
        for column in prompt_columns:

            for batch in tqdm(batch_iterable(dataset), desc=f"Evaluating {column}"):
                batch_hash = [queue._hash_message(row[column]) for row in batch]
                results = await queue.get_batch_results(batch_hash)

                for i, (hash, row) in enumerate(zip(batch_hash, batch)):
                    hash_result = results.get(hash, None)

                    if hash_result:
                        sample_res = {
                            "model_translation": model_translation,
                            "ground_truth": ground_truth,
                            "hash": hash,
                            "translation_direction": hash_result['translation_direction'],
                            "prompt_type": hash_result['prompt_type'],
                            "prompt": row[column]
                        }

                        ground_truth = row['translation'][hash_result['translation_direction'].split('_')[-1]]
                        model_translation = hash_result['result']

                        for metric in METRIC_MAPPING:
                            results = METRIC_MAPPING[metric].compute(predictions=[model_translation], references=[[ground_truth]])
                            sample_res[metric] = results

                        samples[column].append(sample_res)

    with open('samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

   
if __name__ == "__main__":
    asyncio.run(main())