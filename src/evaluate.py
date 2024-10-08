import wandb
import lm_eval
import logging
import pandas as pd
from args import load_config
from lm_eval.loggers import WandbLogger
from utils import build_model_input_string, generate_lang_task_list, log_eval_samples


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


METRICS = [ 'exact_match', 'f1', 'acc', 'f1', 'acc_stderr']

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
    task_list, lang_2_task = generate_lang_task_list(config.task)

    print(f"Task list: {task_list}")

    results = lm_eval.simple_evaluate(
        model=config.model.model_type,
        model_args=input_model_string,
        tasks=task_list,
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

    metrics_list = []
    for lang, tasks in lang_2_task.items():

        for task in tasks:
            lang_metrics = {'lang': lang, 'task': task}
            for metric, value in metric_results[task].items():
                if metric != 'alias':
                    lang_metrics[metric.replace(',none', '').replace(',remove_whitespace','')] = value
            
            metrics_list.append(lang_metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    grouped_metrics_df = metrics_df.groupby('lang').agg({metric: 'mean' for metric in METRICS if metric in metrics_df.columns}).reset_index()

    metrics_config_dict = {}

    for _, rows in grouped_metrics_df.iterrows():
        lang = rows['lang']
        for metric in METRICS:
            if metric in rows:
                metrics_config_dict[f"{lang}_{metric}"] = rows[metric]

    # # Log the results to wandb as metrics
    config_dict = {k: v for k, v in results['config'].items() if isinstance(v, (str, int))}
    config_dict['num_fewshot'] = config.eval.num_fewshot


    # if config.eval.limit is None:

    run_name = generate_wandb_run_name(config_dict['model_args'], config.eval.num_fewshot)
    with wandb.init( project=config.task.wandb_project, job_type=config.task.wandb_job_type, name=run_name) as run:
        run.log(metrics_config_dict)
        run.log(config_dict)

        all_metrics = wandb.Table(dataframe=metrics_df)
        run.log({"Results": all_metrics})

        # Use LM Eval wandb logger to log the samples
        log_eval_samples(run, results)


if __name__ == "__main__":
    main()
