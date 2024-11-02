import json
import os

import wandb
import lm_eval
import logging
import pandas as pd
from lm_eval.loggers.utils import _handle_non_serializable

from args import load_config
from utils import build_model_input_string, generate_lang_task_list, log_eval_samples

lm_eval_logger = logging.getLogger("lm_eval")
lm_eval_logger.setLevel(logging.ERROR)

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

METRICS = [ 'acc', 'exact_match', 'f1', 'acc_stderr', 'ter', 'bleu', 'chrf']


def generate_wandb_run_name(model_args: str, num_few_shot: int):
    model_args = model_args.replace('pretrained=', '')
    org_model_name = model_args.split(',')[0]

    model_name = org_model_name.split('/')[-1].replace('_', '-')

    return f"{model_name}-{num_few_shot}shot"


def main():
    config = load_config()
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Batch size: {config.eval.batch_size}")
    logger.info(f"Random seed: {config.eval.random_seed}")
    logger.info(f"Use cache: {config.eval.use_cache}")
    logger.info(f"num_fewshot: {config.eval.num_fewshot}")

    input_model_string = build_model_input_string(config.model)
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
    
    for task_or_task_list in task_list:
        # Filter out tasks we already have results for
        if metrics_df is not None:
            if isinstance(task_or_task_list, list):
                task_or_task_list = [task for task in task_or_task_list if task not in metrics_df["task"].unique()]
            elif isinstance(task_or_task_list, str) and task_or_task_list in metrics_df["task"].unique():
                task_or_task_list = []

        if not task_or_task_list:
            continue

        results = lm_eval.simple_evaluate(
            model=config.model.model_type,
            model_args=input_model_string,
            tasks=task_or_task_list,
            log_samples=config.eval.log_samples,
            num_fewshot=config.eval.num_fewshot if config.eval.num_fewshot > 0 else None,
            batch_size=config.eval.batch_size,
            max_batch_size=config.eval.max_batch_size,
            random_seed=config.eval.random_seed,
            numpy_random_seed=config.eval.numpy_random_seed,
            torch_random_seed=config.eval.torch_random_seed,
            fewshot_random_seed=config.eval.fewshot_random_seed,
            write_out=config.eval.write_out,
            limit=config.eval.limit,
        )

        configs.update(results["configs"])
        json.dump(configs, open(configs_path, "w"), ensure_ascii=False, indent=4)

        samples.update(results["samples"])
        json.dump(
            samples,
            open(samples_path, "w"),
            ensure_ascii=False,
            indent=4,
            default=_handle_non_serializable
        )

        metric_results.update(results['results'])
        json.dump(metric_results, open(metric_results_path, "w"), ensure_ascii=False, indent=4)

        metrics_list = []
        completed_tasks = [task_or_task_list] if isinstance(task_or_task_list, str) else task_or_task_list
        
        for task in completed_tasks:
            task_and_language = task.split("_prompt")[0]
            language = task_and_language.replace(f"{config.task.task_name}_", "")

            lang_metrics = {"lang": language, "task": task}

            for metric, value in metric_results[task].items():
                if metric != "alias":
                    lang_metrics[metric.replace(',none', '').replace(',remove_whitespace','')] = value
            
            metrics_list.append(lang_metrics)

        if metrics_df is None:
            metrics_df = pd.DataFrame(metrics_list)
        else:
            metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_list)], ignore_index=True)
        
        metrics_df.to_json(metric_df_path, orient="records", lines=True, force_ascii=False)

    metrics_config_dict = {}
    grouped_metrics_df = metrics_df.groupby('lang').agg(
        {metric: 'mean' for metric in METRICS if metric in metrics_df.columns}
    ).reset_index()

    for _, rows in grouped_metrics_df.iterrows():
        lang = rows['lang']
        for metric in METRICS:
            if metric in rows:
                metrics_config_dict[f"{lang}_{metric}"] = rows[metric]
<<<<<<< HEAD

=======
    
>>>>>>> 7aefd68 (split evaluations by task)
    # # Log the results to wandb as metrics
    config_dict = {k: v for k, v in results['config'].items() if isinstance(v, (str, int))}
    config_dict['num_fewshot'] = config.eval.num_fewshot

    all_results = {
        "samples": samples,
        "configs": configs,
        "results": metric_results,
        "config": results["config"]
    }

    run_name = generate_wandb_run_name(config_dict['model_args'], config.eval.num_fewshot)
    with wandb.init(project=config.task.wandb_project, job_type=config.task.wandb_job_type, name=run_name) as run:
        run.log(metrics_config_dict)
        run.log(config_dict)

        all_metrics = wandb.Table(dataframe=metrics_df)
        run.log({"Results": all_metrics})

        # Use LM Eval wandb logger to log the samples
        log_eval_samples(run, all_results)


if __name__ == "__main__":
    main()
