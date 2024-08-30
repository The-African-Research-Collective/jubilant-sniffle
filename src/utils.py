import os
import logging

from args import ModelConfig, TaskConfig
from typing import List

logger = logging.getLogger(__name__)
REQUIRED_ENV_VARS = ['CUDA_VISIBLE_DEVICES']

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
    for lang in task_config.languages:
        lang_task_list.append(task_config.task_template.replace("{{language}}", lang))
    return lang_task_list