import os
import json
import wandb
import lm_eval
import logging
import pandas as pd
import numpy as np
from args import load_config
from utils import build_model_input_string, generate_lang_task_list


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


METRICS = [ 'acc', 'f1', 'acc_stderr']


def main():
    config = load_config()
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Batch size: {config.eval.batch_size}")
    logger.info(f"Random seed: {config.eval.random_seed}")
    logger.info(f"Use cache: {config.eval.use_cache}")
    logger.info(f"num_fewshot: {config.eval.num_fewshot}")

    input_model_string = build_model_input_string(config.model)
    task_list, lang_2_task = generate_lang_task_list(config.task)
    print(f"task list: {task_list}\nlang_2_task: {lang_2_task}")

    print(f"Task list: {task_list}")

    print(f"device: {config.eval.log_samples}")

    results = lm_eval.simple_evaluate(
        model=config.model.model_type,
        model_args=input_model_string,
        tasks=task_list,
        device=config.eval.device,
        log_samples=config.eval.log_samples,
        num_fewshot=config.eval.num_fewshot if config.eval.num_fewshot > 0 else None,
        batch_size=config.eval.batch_size,
        max_batch_size=config.eval.max_batch_size,
        random_seed=config.eval.random_seed,
        numpy_random_seed=config.eval.numpy_random_seed,
        torch_random_seed=config.eval.torch_random_seed,
        fewshot_random_seed=config.eval.fewshot_random_seed,
        write_out=config.eval.write_out,
        limit=config.eval.limit
    )

    metric_results = results['results']
    print(metric_results)

    metrics_list = []
    for lang, tasks in lang_2_task.items():

        for task in tasks:
            lang_metrics = {'lang': lang, 'task': task}
            for metric, value in metric_results[task].items():
                if metric != 'alias':
                    lang_metrics[metric.replace(',none', '')] = value
            
            metrics_list.append(lang_metrics)
    print(f"metric list {metrics_list}")
    metrics_df = pd.DataFrame(metrics_list)

    results_dir = f"../results/{config.task.task_name}/{config.eval.num_fewshot}/" if config.eval.num_fewshot > 0 else f"../results/{config.task.task_name}/0"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"{config.model.model_name.split('/')[-1]}.csv"

    metrics_df.to_csv(f'{results_dir}/{filename}', index=False)

    metrics_df = metrics_df.groupby('lang').agg({ metric :'mean' for metric in METRICS }).reset_index()

    metrics_config_dict = {}

    for _, rows in metrics_df.iterrows():
        lang = rows['lang']
        for metric in METRICS:
            metrics_config_dict[f"{lang}_{metric}"] = rows[metric]
    print(f"metrics config list: {metrics_config_dict}")
    print(f"result configs: {results['config']}")
    # # Log the results to wandb as metrics
    config_dict = {k: v for k, v in results['config'].items() if isinstance(v, (str, int))}
    print(f"config dict: {config_dict}")
    config_dict['num_fewshot'] = config.eval.num_fewshot

    # wandb.init( project=config.task.wandb_project, job_type=config.task.wandb_job_type)
    
    # wandb.log(metrics_config_dict)
    # wandb.log(config_dict)


if __name__ == "__main__":
    main()
