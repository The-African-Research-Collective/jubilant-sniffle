#!/bin/bash
# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=160:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:jimmygpu:1
#SBATCH -o slurm_logs/JOB%j.out
#SBATCH -e slurm_logs/JOB%j-err.out

# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL)

#SBATCH --mail-type=ALL

# set -e;

{
    export WANDB_ENTITY="african-research-collective"
    export WANDB_CACHE_DIR="/jarmy/.cache/wandb"
    export WANDB_DATA_DIR="/jarmy/.wandb"

    task="masakhanews"
    model_path=configs/models
    batch_size=8

    export CUDA_VISIBLE_DEVICES=3
    export TRUST_REMOTE_CODE=True

    single_gpu_models=(
        "afriteva_v2_large_ayaft"
        # "jacaranda_afrollama"
        # "lelapa_inkuba_0_4b"
        # "google_gemma-1_7b_it"
        # "meta_llama_8b_instruct"
        # "ubc_nlp_cheetah_base"
        # "google_gemma-2-9b-it"
        # "meta_llama-2_7b_chat"
        # "google_gemma-2_27b_it"
        # "llamax3-8b-alpaca"
        # "bigscience_mt0_large"
        # "bigscience_mt0_small"
        # "google_flan_t5_small"
        # "google_flan_t5_base"
        # "google_flan_t5_large"
        # "bigscience_mt0_xl"
        # "bigscience_mt0_base"
        # "bigscience_bloomz_1b7"
        # "bigscience_bloomz_3b"
        # "bigscience_bloomz_7b1"
        # "meta_llama_3-2_1b_instruct"
    )

    multi_gpu_models=(
        # "aya_101"
        # "bigscience_mt0_xxl"
        # "meta-llama-3-70b-instruct"
        # "bigscience_bloomz"
    )

    all_models=("${single_gpu_models[@]}" "${multi_gpu_models[@]}")
    task_config=configs/tasks/$task.yaml

    mkdir -p task_logs/$task

    for num_fewshot_samples in 5
    do
        for model in "${all_models[@]}"
        do
            if [ ! -f "task_logs/$task/model" ]; then
                echo "Running model: $model"

                python src/evaluate.py --model-config-yaml "${model_path}/${model}.yaml" \
                --task-config-yaml ${task_config} \
                --eval.num-fewshot ${num_fewshot_samples} \
                --eval.batch-size ${batch_size} &> "logs/masakhane_${model}-${num_fewshot_samples}-shot.log" && \
                touch "task_logs/$task/model-$num_fewshot_samples-samples"
            else
                echo "Skipping the $num_fewshot_samples for $model. May have already been run"
            fi
        done
    done
}