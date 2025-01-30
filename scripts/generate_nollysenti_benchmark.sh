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

NUM_FEWSHOT=$1 # Number of fewshot samples

task_config=configs/tasks/nollysenti.yaml
model_path=configs/models
batch_size=4
task="nollysenti"

declare -a models

models=(
    "meta_llama_8b_instruct"
    "meta_llama_3_1_8b_instruct.yaml"
    "meta_llama_70b_instruct"
    "meta_llama-2_7b_chat"
    "lelapa_inkuba_0_4b"
    "bigscience_mt0_xl"
    "bigscience_mt0_xxl"
    "google_gemma-1_7b_it"
    "google_gemma-2_27b_it"
    "jacaranda_afrollama"
    "aya_101"
    "llamax_8b"
    "afriteva_v2_large_ayaft"
)


for num_fewshot_samples in ${NUM_FEWSHOT}
do 
    for model in "${models[@]}"
    do
        echo "Running model: $model"
        mkdir -p runs/${task}/${model}

        python src/evaluate.py --model-config-yaml ${model_path}/${model}.yaml \
        --task-config-yaml ${task_config} \
        --eval.num-fewshot ${num_fewshot_samples} \
        --eval.batch-size ${batch_size} \
        --run-dir runs/${task}/${model}
    done
done
