#!/bin/bash

#==============================================================================#
# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --nodelist=watgpu208
#SBATCH --time=48:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3

# Set output file destinations (optional)
# By default, output will appear in a file in the submission directory:
# slurm-$job_number.out
# This can be changed:
#SBATCH -o slurm_logs/JOB%j.out # File to which STDOUT will be written
#SBATCH -e slurm_logs/JOB%j-err.out # File to which STDERR will be written

# email notifications: Get email when your job starts, stops, fails, completes...
# Set email address
#SBATCH --mail-user=ogundepoodunayo@gmail.com
# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL):
#SBATCH --mail-type=ALL
#==============================================================================#

{
    export WANDB_ENTITY="african-research-collective"
    
    task="masakhanews"
    model_path=configs/models
    batch_size=auto

    single_gpu_models=(
        # "meta_llama-2_7b_chat"
        # "meta_llama_8b_instruct"
        # "lelapa_inkuba_0_4b"
        # "google_gemma-1_7b_it"
        # "jacaranda_afrollama"
        # "llamax3-8b-alpaca"
        # "google_gemma_2-9b-it"
        # "google_gemma-2_27b_it"
        # "meta_llama_3-1_8b_instruct"
    )

    not_urgent_models=(
        "afriteva_v2_large_ayaft"
        # "ubc_nlp_cheetah_base"
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
        # "aya-101"
        # "meta-llama_3-1_70b_instruct"
        # "bigscience_mt0_xxl"
        # "bigscience_bloomz"
    )

    all_models=("${single_gpu_models[@]}" "${multi_gpu_models[@]}" "${not_urgent_models[@]}")
    task_config=configs/tasks/$task.yaml

    mkdir -p task_logs/$task

    for num_fewshot_samples in 0
    do
        for model in "${all_models[@]}"
        do
            if [ ! -f "task_logs/$task/$model-$num_fewshot_samples-samples" ]; then
                echo "Running model: $model"

                python src/evaluate.py --model-config-yaml "${model_path}/${model}.yaml" \
                --task-config-yaml "${task_config}" \
                --run-dir "runs/$task/$model" \
                --eval.num-fewshot ${num_fewshot_samples} \
                --eval.split-tasks True \
                --eval.batch-size ${batch_size} &> "logs/${task}_${model}-${num_fewshot_samples}-shot.log" && \
                touch "task_logs/$task/$model-$num_fewshot_samples-samples"
            else
                echo "Skipping the $num_fewshot_samples for $model. May have already been run"
            fi
        done
    done
}