#!/bin/bash
# Set resource requirements: Queues are limited to seven day allocations
# Time format: HH:MM:SS
#SBATCH --time=160:00:00
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=JIMMY
#SBATCH -o slurm_logs/JOB%j.out
#SBATCH -e slurm_logs/JOB%j-err.out

# Set types of notifications (from the options: BEGIN, END, FAIL, REQUEUE, ALL)
#SBATCH --mail-user=akin.o.oladipo@gmail.com
#SBATCH --mail-type=ALL

{
    export WANDB_ENTITY="african-research-collective"
    
    task="mafand"
    model_path=configs/models
    batch_size=auto
    
    export TRUST_REMOTE_CODE=True
    
    single_gpu_models=(
        # "lelapa_inkuba_0_4b"
        # "google_gemma-1_7b_it"
        # "meta_llama-2_7b_chat"
        "meta_llama_8b_instruct"
        # "jacaranda_afrollama"
        # "llamax3-8b-alpaca"
        # "meta_llama_3-1_8b_instruct"
        # "google_gemma_2-9b-it"
        # "google_gemma-2_27b_it"
    )

    not_urgent_models=(
        # "afriteva_v2_large_ayaft"
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