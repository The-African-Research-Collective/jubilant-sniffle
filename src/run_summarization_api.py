import os
import json
import asyncio
import wandb
import logging
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer
from string import Template
from copy import copy
from args import load_config
from utils import generate_lang_task_list
from processing_queue import SummarizationQueue
from tqdm import tqdm
from collections import defaultdict
import evaluate

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

TOGETHER2HGF = {
    "meta-llama/Llama-3-8b-chat-hf": "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-9b-it": "google/gemma-2-9b-it",
    "google/gemma-2-27b-it": "google/gemma-2-9b-it",
    "meta-llama/Llama-3-70b-chat-hf": "meta-llama/Meta-Llama-3-8B-Instruct",
}


def check_input_tokens(input_text, max_tokens, tokenizer):

    input_tokens = tokenizer(input_text)['input_ids']
    if len(input_tokens) > max_tokens:
        # truncate input text
        input_text = tokenizer.decode(input_tokens[:max_tokens])
    
    return input_text


def process_xlsum(example, prompts, language, tokenizer, max_tokens=7500):

    def format_prompt(prompt, source_text, language):
        return prompt.substitute(text=source_text, language=language)

    for prompt_file in prompts:
        source_text = example['text']
        source_text = check_input_tokens(source_text, max_tokens, tokenizer)
        example[f"{prompt_file}"] = format_prompt(copy(prompts[prompt_file]), source_text, language)
    
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
    
    queue = SummarizationQueue(db_path=os.path.join(config.run_dir, f"{config.task.task_name}_{config.model.model_name.replace("/", "_")}.db"), batch_size=32)
    tokenizer = AutoTokenizer.from_pretrained(TOGETHER2HGF[config.model.model_name])

    def batch_iterable(iterable, n=32):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]
    
    processed_datasets = []

    for task in config.task.languages:
        task_dataset = load_dataset(config.task.hub_dataset, task)
        task_dataset = task_dataset[config.task.evaluation_split]
        initial_columns = set(task_dataset.column_names)

        task_dataset = task_dataset.map(process_xlsum, fn_kwargs={'language': task, 'prompts': prompts, 'tokenizer': tokenizer})
        prompt_columns = set(task_dataset.column_names) - set(initial_columns)

        processed_datasets.append((task_dataset, prompt_columns))

        for column in prompt_columns:
            for text_batch in tqdm(batch_iterable(task_dataset[column]), desc=f"Processing {column}"):
                language = task
                prompt_type = column

                await queue.enqueue(text_batch, language, prompt_type)

        stats = await queue.process_queue(model_name=config.model.model_name)
        logger.info(f"Run Stats: {stats}")

    




if __name__ == "__main__":
    asyncio.run(main())