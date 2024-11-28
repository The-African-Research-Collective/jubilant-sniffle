import os
import logging
import pandas as pd

from args import ModelConfig, TaskConfig
from typing import List, Dict, Any
from collections import defaultdict
from lm_eval.loggers import WandbLogger


logger = logging.getLogger(__name__)
REQUIRED_ENV_VARS = ['CUDA_VISIBLE_DEVICES']
MAFAND_CODE_2_LANG = {
    "en": "English",
    "amh": "Amharic",
    "fr": "French",
    "bam": "Bambara",
    "lug": "Luganda",
    "luo": "Luo",
    "kin": "Kinyarwanda",
    "tsn": "Setswana",
    "pcm": "Nigerian Pidgin",
    "zul": "Zulu",
    "swa": "Swahili",
    "hau": "Hausa",
    "ibo": "Igbo",
    "yor": "Yoruba",
    "mos": "Mossi",
    "nya": "Chichewa",
    "bbj": "Ghomala",
    "wol": "Wolof",
    "twi": "Twi",
    "fon": "Fon",
    "sna": "Shona",
    "xho": "Xhosa",
    "ewe": "Ewe",
}




def check_env_file_and_vars(env_file='.env'):
    # Check if the .env file exists
    if not os.path.isfile(env_file):
        raise FileNotFoundError(f"The environment file {env_file} does not exist.")

    # Load the environment variables from the file
    with open(env_file, 'r') as file:
        env_vars = [line.split('=')[0] for line in file.readlines() if '=' in line]

    # Check if all required environment variables are set
    missing_vars = [var for var in REQUIRED_ENV_VARS if var not in env_vars]
    if missing_vars:
        raise EnvironmentError(f"The following required environment variables are missing: {', '.join(missing_vars)}")

    logger.info(f"All required environment variables are present in {env_file}.")

def build_model_input_string(model_args: ModelConfig):
    """
    Builds a string representation of the model input based on the provided ModelConfig.
    Args:
        model_args (ModelConfig): The configuration object containing the model arguments.
    Returns:
        str: The string representation of the model input.
    """

    output_string = "pretrained=" + model_args.model_name + ","

    if model_args.trust_remote_code:
        output_string += "trust_remote_code=True,"
    
    if model_args.parallelize:
        output_string += "parallelize=True,"
    
    if model_args.add_bos_token:
        output_string += "add_bos_token=True,"
    
    if output_string.endswith(","):
        output_string =  output_string[:-1]
    
    return output_string


def generate_lang_task_list(task_config: TaskConfig) -> List[str]:
    """
    Generate a list of language-specific tasks based on the given task configuration.

    Args:
        task_config (TaskConfig): The task configuration object containing the list of languages and the task template.

    Returns:
        List[str]: A list of language-specific tasks.

    """
    lang_task_list = []
    language_2_task = defaultdict(list)

    templates = task_config.task_template.split(',')

    for template in templates:
        for lang in task_config.languages:
            lang_task_list.append(template.replace("{{language}}", lang))
            language_2_task[lang].append(template.replace("{{language}}", lang))
    
    return lang_task_list, language_2_task

def _get_config(results) -> Dict[str, Any]:
    """Get configuration parameters."""
    task_configs = results.get("configs", {})
    cli_configs = results.get("config", {})
    configs = {
        "task_configs": task_configs,
        "cli_configs": cli_configs,
    }

    return configs


def log_eval_samples(run, results, task_names: list[str] = None):
    """Log evaluation samples to W&B.

    Args:
        samples (Dict[str, List[Dict[str, Any]]]): Evaluation samples for each task.
    """
    samples = results["samples"]

    if task_names is None:
        task_names: List[str] = list(results.get("results", {}).keys())
    
    group_names: List[str] = list(results.get("groups", {}).keys())
    configs = _get_config(results)

    task_names: List[str] = [
        x for x in task_names if x not in group_names
    ]

    ungrouped_tasks = []
    tasks_by_groups = {}

    for task_name in task_names:
        group_names = configs['task_configs'][task_name].get("group", None)
        
        if group_names:
            if isinstance(group_names, str):
                group_names = [group_names]

            for group_name in group_names:
                if not tasks_by_groups.get(group_name):
                    tasks_by_groups[group_name] = [task_name]
                else:
                    tasks_by_groups[group_name].append(task_name)
        else:
            ungrouped_tasks.append(task_name)

    for task_name in ungrouped_tasks:
        eval_preds = samples[task_name]

        # log the samples as a W&B Table
        df = WandbLogger()._generate_dataset(eval_preds, configs['task_configs'].get(task_name))
        run.log({f"{task_name}_eval_results": df})

    for group, grouped_tasks in tasks_by_groups.items():
        grouped_df = pd.DataFrame()
        
        for task_name in grouped_tasks:
            eval_preds = samples[task_name]
            df = WandbLogger()._generate_dataset(
                eval_preds, configs['task_configs'].get(task_name)
            )
            df["group"] = group
            df["task"] = task_name
            grouped_df = pd.concat([grouped_df, df], ignore_index=True)

        run.log({f"{group}_eval_results": grouped_df})
