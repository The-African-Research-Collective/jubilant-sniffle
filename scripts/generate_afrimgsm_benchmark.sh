#!/bin/bash

export HF_HOME='~/scratch/.cache/huggingface'
export HF_DATASETS_CACHE='~/scratch/.cache/huggingface'

task_config=configs/tasks/afrimgsm.yaml
model_path=configs/models
batch_size='auto'

declare -a models
export CUDA_VISIBLE_DEVICES=0,1

models=(
    "jacaranda_afrollama_vllm"
    "google_gemma-1_7b_it_vllm"
    # "meta_llama-3.1_70b_instruct"
    # "afriteva_v2_large_ayaft"
    # "llamax3-8B-Alpaca"
    # "aya_101"
    # "meta_llama_8b_instruct"
    # "meta_llama-2_7b_chat"
    # "google_gemma-1_7b_it"
    # "google_gemma-2_9b_it"
#     "lelapa_inkuba_0_4b"
    # "google_gemma-2_27b_it"
    # "jacaranda_afrollama"
    # "meta_llama-3.1_8b_instruct"
)

for num_fewshot_samples in 0
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
