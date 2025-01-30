import os
import json
import asyncio
import logging
import evaluate
from itertools import product
from datasets import load_dataset
from copy import copy
from string import Template
from tqdm import tqdm
from collections import defaultdict

from args import load_config
from processing_queue import SummarizationQueue
from transformers import AutoTokenizer

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

TRANSLATION_METRICS = ['bleu', 'rouge', 'bertscore']
METRIC_MAPPING = {
    metric: evaluate.load(metric) for metric in TRANSLATION_METRICS
}
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

def batch_iterable(iterable, n=32):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

async def main():

    config = load_config()
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Batch size: {config.eval.batch_size}")
    logger.info(f"Random seed: {config.eval.random_seed}")
    logger.info(f"Use cache: {config.eval.use_cache}")
    logger.info(f"num_fewshot: {config.eval.num_fewshot}")
    
    queue = SummarizationQueue(db_path=os.path.join(config.run_dir, f"{config.task.task_name}_{config.model.model_name.replace("/", "_")}.db"), batch_size=32)
    tokenizer = AutoTokenizer.from_pretrained(TOGETHER2HGF[config.model.model_name])

    prompt_types = await queue.get_all_prompt_types()
    prompt_types = [prompt_type[0] for prompt_type in prompt_types]

    print(f"Prompt types: {prompt_types}")

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
    

    for task in config.task.languages:

        task_dataset = load_dataset(config.task.hub_dataset, task)
        task_dataset = task_dataset[config.task.evaluation_split]
        initial_columns = set(task_dataset.column_names)

        task_dataset = task_dataset.map(process_xlsum, fn_kwargs={'language': task, 'prompts': prompts, 'tokenizer': tokenizer})
        prompt_columns = set(task_dataset.column_names) - set(initial_columns)

        for column in prompt_columns:
            samples = defaultdict(list)

            for text_batch, summaries in tqdm(zip(batch_iterable(task_dataset[column]), batch_iterable(task_dataset["summary"]) ),
                                desc=f"Processing prompt type: {column}"):
                batch_hash = [queue._hash_message(text) for text in text_batch ]
                summaries = [summaries[i] for i in range(len(summaries))]
                batch_samples  = []

                results = await queue.get_batch_results(batch_hash)

                for i, (hash, summary) in enumerate(zip(batch_hash, summaries)):
                    hash_result = results.get(hash, None)
                    sample_results = {}
                    if hash_result:
                        for metric in METRIC_MAPPING:
                            try:
                                if metric == 'bertscore':
                                    score = METRIC_MAPPING[metric].compute(predictions=[hash_result['summary']], references=[[summary]], model_type='microsoft/mdeberta-v3-base')
                                else:
                                    score = METRIC_MAPPING[metric].compute(predictions=[hash_result['summary']], references=[[summary]])
                                sample_results[metric] = score
                            except Exception as e:
                                logger.error(f"Error computing metric {metric} for hash {hash}: {e}")
                    else:
                        raise ValueError(f"Hash {hash} not found in results")

                    batch_samples.append({
                        "generated_summary": hash_result['summary'],
                        "groundtruth_summary": summary,
                        "language": task,
                        "prompt_type": column,
                        "metrics": sample_results
                    })

                samples[column].extend(batch_samples)
    
            with open(f'{config.run_dir}/results/{config.task.task_name}_{task}_{config.model.model_name.replace("/", "_")}_samples_{column}.json', 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    asyncio.run(main())