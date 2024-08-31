import os
import lm_eval
import logging
from args import load_config
from utils import build_model_input_string, generate_lang_task_list
from lm_eval.loggers import WandbLogger

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)




def main():
    config = load_config()
    
    logger.info(f"Model: {config.model}")
    logger.info(f"Task: {config.task}")
    logger.info(f"Batch size: {config.eval.batch_size}")
    logger.info(f"Random seed: {config.eval.random_seed}")
    logger.info(f"Use cache: {config.eval.use_cache}")
    logger.info(f"num_fewshot: {config.eval.num_fewshot}")


    input_model_string = build_model_input_string(config.model)
    task_list = generate_lang_task_list(config.task)


    results = lm_eval.simple_evaluate(
        model=config.model.model_type,
        model_args=input_model_string,
        tasks=task_list,
        log_samples=config.eval.log_samples,
        num_fewshot=config.eval.num_fewshot,
        batch_size=config.eval.batch_size,
        max_batch_size=config.eval.max_batch_size,
        random_seed=config.eval.random_seed,
        numpy_random_seed=config.eval.numpy_random_seed,
        torch_random_seed=config.eval.torch_random_seed,
        fewshot_random_seed=config.eval.fewshot_random_seed,
        write_out=config.eval.write_out,
        limit=config.eval.limit
    )
    
    if config.eval.limit != -1 and not config.eval.write_out:
        wandb_logger = WandbLogger(
            project=config.task.wandb_project,
            job_type=config.task.wandb_job_type
        )
        wandb_logger.post_init(results)
        wandb_logger.log_eval_result()
        wandb_logger.log_eval_samples(results["samples"])

if __name__ == "__main__":
    main()
