#!/bin/bash
    
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

# source llm_evaluation/bin/activate

HF_HOME=/jarmy/odunayoogundepo/jubilant-sniffle/cache
TRANSFORMERS_CACHE=${HF_HOME}


task_config=configs/tasks/afrimmlu-direct.yaml
model_path=configs/models
batch_size=2

declare -a models
export CUDA_VISIBLE_DEVICES=0,3
export TRUST_REMOTE_CODE=True
HF_HOME=/jarmy/odunayoogundepo/jubilant-sniffle/cache
TRANSFORMERS_CACHE=${HF_HOME}
WANDB_DIR=${HF_HOME}
TMPDIR=/jarmy/odunayoogundepo/jubilant-sniffle/tmp

models=(
    # "afriteva_v2_large_ayaft"
    # "meta_llama_8b_instruct"
    # "meta_llama_70b_instruct"
    # "meta_llama_3_1_8b_instruct.yaml"
    # "meta_llama-2_7b_chat"
    # "aya_101"
    # "lelapa_inkuba_0_4b"
    # "bigscience_mt0_xl"
    "bigscience_mt0_xxl"
    # "google_gemma-1_7b_it"
    # "google_gemma-2_27b_it"
    # "jacaranda_afrollama"
    # "llamax_8b"
)

for num_fewshot_samples in 5
do 
    for model in "${models[@]}"
    do
        echo "Running model: $model"

        python src/evaluate.py --model-config-yaml ${model_path}/${model}.yaml \
            --task-config-yaml ${task_config} \
            --eval.num-fewshot ${num_fewshot_samples} \
            --eval.batch-size ${batch_size} 
    done
done
