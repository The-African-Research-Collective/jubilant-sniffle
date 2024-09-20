num_fewshot_samples=$1
task_config=configs/tasks/afrimmlu-translate.yaml
model_path=configs/models
batch_size=8

declare -a models
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

models=(
    
)

models=(
    "afriteva_v2_large_ayaft"
    "meta_llama_8b_instruct"
    "aya_101"
    "bigscience_bloomz_1b7"
    "bigscience_bloomz_3b"
    "bigscience_bloomz_7b1"
    "lelapa_inkuba_0_4b"
    "bigscience_mt0_base"
    "bigscience_mt0_large"
    "bigscience_mt0_small"
    "bigscience_mt0_xl"
    "google_flan_t5_base"
    "google_flan_t5_large"
    "google_flan_t5_small"
)

for model in "${models[@]}"
do
    echo "Running model: $model"

    python src/evaluate.py --model-config-yaml ${model_path}/${model}.yaml \
     --task-config-yaml ${task_config} \
     --eval.num-fewshot ${num_fewshot_samples} \
     --eval.batch-size ${batch_size}
done
