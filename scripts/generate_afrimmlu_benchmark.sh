num_fewshot_samples=$1
task_config=configs/tasks/afrimmlu-direct.yaml
model_path=configs/models
batch_size=8

declare -a models
export CUDA_VISIBLE_DEVICES=1,2,3,4,5

models=(
    "afriteva_v2_large_ayaft"
    "meta_llama_8b_instruct"
)

for model in "${models[@]}"
do
    echo "Running model: $model"

    python src/evaluate.py --model-config-yaml ${model_path}/${model}.yaml \
     --task-config-yaml ${task_config} \
     --eval.num-fewshot ${num_fewshot_samples} \
     --eval.batch-size ${batch_size}
done
