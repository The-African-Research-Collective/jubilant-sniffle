#!/bin/bash
    
# To be submitted to the SLURM queue with the command:
# sbatch batch-submit.sh

# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=20:00:00
#SBATCH --mem=10GB
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
#SBATCH --nodelist=watgpu308

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

# source llm_evaluation/bin/activate


task_config=configs/tasks/sib_contd.yaml
model_path=configs/models
batch_size=auto
task="sib"

mkdir -p task_logs/$task

declare -a models
export CUDA_VISIBLE_DEVICES=0,1,2
export TRUST_REMOTE_CODE=True
# HF_HOME=/jarmy/odunayoogundepo/jubilant-sniffle/cache
# TRANSFORMERS_CACHE=${HF_HOME}
# WANDB_DIR=${HF_HOME}

models=(
    # "afriteva_v2_large_ayaft"
    "meta_llama_8b_instruct"
    "meta_llama_70b_instruct"
    "meta_llama_3_1_8b_instruct"
    "meta_llama-2_7b_chat"
    "aya_101"
    # "lelapa_inkuba_0_4b"
    # "bigscience_mt0_xl"
    "bigscience_mt0_xxl"
    "google_gemma-1_7b_it"
    "google_gemma-2_27b_it"
    "google_gemma_2-9b-it"
    "jacaranda_afrollama"
    "llamax_8b"
    "meta_llama_3_2_1b_instruct"
)


for num_fewshot_samples in 0
do 
    for model in "${models[@]}"
    do
        echo "Running model: $model"

        python src/evaluate.py --model-config-yaml ${model_path}/${model}.yaml \
            --task-config-yaml ${task_config} \
            --eval.num-fewshot ${num_fewshot_samples} \
            --eval.batch-size ${batch_size} \
            --run-dir runs/${task}/${model}
    done
done
