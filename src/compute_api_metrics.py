import os
import json
import asyncio
import logging
import evaluate
from typing import List
from itertools import product
from datasets import load_dataset
from copy import copy
from string import Template
from tqdm import tqdm
from collections import defaultdict

from args import load_config
from processing_queue import MessageQueue
from utils import MAFAND_CODE_2_LANG

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

TRANSLATION_METRICS = ['ter', 'bleu', 'chrf']

METRIC_MAPPING = {
    metric: evaluate.load(metric) for metric in TRANSLATION_METRICS
}

def process_translation_example_prompt_mafand(example, language_pair, prompts, fewshot_samples=None):

    def format_prompt(prompt, source_sentence, source_language, target_language, fewshot_prompt=None):
        if fewshot_prompt:
            return prompt.substitute(source_sentence=source_sentence,
                        source_language=source_language,
                        target_language=target_language,
                        fewshot_samples=fewshot_prompt)
        
        return prompt.substitute(source_sentence=source_sentence, source_language=source_language, target_language=target_language)
    
    source_code = language_pair.split('_')[0]
    target_code = language_pair.split('_')[1]

    for prompt_file in prompts:
        source_sentence = example['translation'][source_code]
        target_sentence = example['translation'][target_code]
        source_language = MAFAND_CODE_2_LANG[source_code]
        target_language = MAFAND_CODE_2_LANG[target_code]

        fewshot_prompt = None

        if fewshot_samples:
            fewshot_prompt = "Here are some examples of translations from the same language pair:\n\n"
            for sample in fewshot_samples['translation']:
                fewshot_prompt += f"{source_language}: {sample[source_code]}\n"
                fewshot_prompt += f"{target_language}: {sample[target_code]}\n\n"
            
            fewshot_prompt += "Inputs:\n\n"

        example[f"{prompt_file}_{source_code}_{target_code}"] = format_prompt(copy(prompts[prompt_file]), source_sentence, source_language, target_language, fewshot_prompt)
        example[f"{prompt_file}_{target_code}_{source_code}"] = format_prompt(copy(prompts[prompt_file]), target_sentence, target_language, source_language, fewshot_prompt)
    
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
    queue = MessageQueue(db_path=os.path.join(config.run_dir, f"{config.task.task_name}_{config.model.model_name.replace("/", "_")}.db")  , batch_size=32)

    prompt_types, translation_directions = await queue.get_all_translation_direction_and_prompt_types()
    prompt_types = [prompt_type[0] for prompt_type in prompt_types]
    translation_directions = [translation_direction[0] for translation_direction in translation_directions]

    print(f"Prompt types: {prompt_types}")
    print(f"Translation directions: {translation_directions}")

    prompt_translation_pairs = list(product(prompt_types, translation_directions))

    print(f"Prompt translation pairs: {prompt_translation_pairs}")

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

        task_dataset = load_dataset(config.task.hub_dataset, task.replace('_', '-'))
        task_dataset = task_dataset[config.task.evaluation_split]
        initial_columns = set(task_dataset.column_names)

        fewshot_dataset = None
        if config.eval.num_fewshot > 0:
            fewshot_dataset = load_dataset(config.task.hub_dataset, task.replace('_', '-'))[config.task.fewshot_split]
            fewshot_dataset = fewshot_dataset.shuffle(seed=config.eval.fewshot_random_seed).select(range(config.eval.num_fewshot))
            fewshot_dataset: List = fewshot_dataset.to_dict()

        task_dataset = task_dataset.map(process_translation_example_prompt_mafand, fn_kwargs={'language_pair': task, 'prompts': prompts, 'fewshot_samples': fewshot_dataset})
        prompt_columns = set(task_dataset.column_names) - set(initial_columns)

        for column in prompt_columns:
            samples = defaultdict(list)
            column = column.split('_')
            translation_directions= '_'.join(column[2:])
            prompt_type = '_'.join(column[:2])

            for text_batch, original_text in tqdm(zip(batch_iterable(task_dataset[f"{prompt_type}_{translation_directions}"]), batch_iterable(task_dataset[f"translation"]) ),
                                desc=f"Processing prompt type: {prompt_type}, translation direction: {translation_directions}"):
                batch_hash = [ queue._hash_message(text) for text in text_batch ]
                target_language = translation_directions.split('_')[1]
                source_language = translation_directions.split('_')[0]
                target_text = [text[target_language] for text in original_text]
                source_text = [text[source_language] for text in original_text]
                batch_samples  = []

                results = await queue.get_batch_results(batch_hash)

                for i, (hash, target_translation) in enumerate(zip(batch_hash, target_text)):
                    hash_result = results.get(hash, None)
                    sample_results = {}
                    if hash_result:
                        for metric in METRIC_MAPPING:
                            try:
                                score = METRIC_MAPPING[metric].compute(predictions=[hash_result['result']], references=[[target_translation]])
                                sample_results[metric] = score
                            except Exception as e:
                                logger.error(f"Error computing metric {metric} for hash {hash}: {e}")
                                logger.error(f"Target translation: {target_translation}")
                                logger.error(f"Source text: {source_text[i]}")
                    else:
                        raise ValueError(f"Hash {hash} not found in results")

                    batch_samples.append({
                        "source_text": source_text[i],
                        "translation_direction": translation_directions,
                        "target_translation": target_translation,
                        "generated_translation": hash_result['result'],
                        "metrics": sample_results
                    })

                samples[f"{prompt_type}_{translation_directions}"].extend(batch_samples)
    
            with open(f'{config.run_dir}/results/{config.task.task_name}_{config.model.model_name.replace("/", "_")}_samples_{prompt_type}_{translation_directions}.json', 'w', encoding='utf-8') as f:
                json.dump(samples, f, indent=4, ensure_ascii=False)



    # for translation_direction, prompt_type in prompt_translation_pairs:
    #     print(f"Processing prompt type: {prompt_type}, translation direction: {translation_direction}")
    #     result = await queue.get_result(prompt_type=prompt_type,
    #                         translation_direction=translation_direction)

    #     print(result[0])
    #     break




if __name__ == "__main__":
    asyncio.run(main())