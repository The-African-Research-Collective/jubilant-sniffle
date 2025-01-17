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

task_config=configs/tasks/post-training-eval/afri-mmlu-math.yaml
model_path=configs/models/post_training
batch_size=1
task=afri-mmlu-math

declare -a models
export CUDA_VISIBLE_DEVICES=7
export TRUST_REMOTE_CODE=True

models=(
#     "meta_llama_8b_base"
#     "persona_llama70b_math_10k_llama_3_8b_instruct"
#     "persona_math_10k_llama_3_8b_base"
    # "persona_math_10k_lugha_llama_8b_wura_math"
    # # "persona_llama70b_math_10k_llama_3_8b_base"
    # "persona_llama70b_math_10k_lughallama_8b_wura_math"
    # "persona_math_10k_llama_3_8b_instruct"
    # "meta_llama_8b_instruct"
    # "lugha_llama_math"
    # "tulu_sft"
    # "grade_school_math_llama_3_8b_base_no_instruction_mask"
    "grade_school_math_llama_3_8b_base_no_instruction_mask_20k"
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
            --run-dir runs/${task}/${model}_
    done
done
