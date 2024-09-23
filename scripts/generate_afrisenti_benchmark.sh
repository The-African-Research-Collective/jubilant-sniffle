task_config=configs/tasks/afrisenti.yaml
model_path=configs/models
batch_size=8

declare -a models
export CUDA_VISIBLE_DEVICES=0,1,2

models=(
    
)

models=(
    "bigscience_mt0_small"
    "bigscience_mt0_xxl"
    "google_gemma-1_7b_it"
    "meta_llama-2_7b_chat"
    "google_gemma-2_9b_it"
    "meta_llama_8b_instruct"
    "aya_101"
    "google_gemma-2_27b_it"
    "meta_llama_70b_instruct"
    "afriteva_v2_large_ayaft"
    "jacaranda_afrollama" 
    # "bigscience_bloomz_1b7"
    # "bigscience_bloomz_3b"
    # "bigscience_bloomz_7b1"
    # "lelapa_inkuba_0_4b"
    # "bigscience_mt0_base"
    # "bigscience_mt0_large" 
    # "bigscience_mt0_xl"
    # "google_flan_t5_base"
    # "google_flan_t5_large"
    # "google_flan_t5_small"
)

for model in "${models[@]}"
do
  echo "Evaluating model: $model"
  for fewshot in 0 5
  do
        python src/evaluate.py --model-config-yaml ${model_path}/${model}.yaml \
        --task-config-yaml ${task_config} \
        --eval.num-fewshot $fewshot \
        --eval.batch-size ${batch_size} \
        --eval.device 'cuda:0' \
        --eval.log-samples True \
        --eval.limit 2
    done
done
