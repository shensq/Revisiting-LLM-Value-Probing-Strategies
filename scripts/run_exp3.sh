#!/bin/sh


# for model_name in "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B-Instruct" "tiiuae/falcon-7b-instruct"  "bigscience/bloomz-7b1" "Qwen/Qwen2.5-3B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
for model_name in "bigscience/bloomz-7b1" "meta-llama/Llama-3.2-3B-Instruct"
do 
    echo "Running model: $model_name"
    modified_model="${model_name//\//-}"
    sbatch --job-name="$modified_model-scoring" scripts/sbatch_action_scoring.sh "$model_name" 
done


# for model_name in "Qwen/Qwen2.5-14B-Instruct"
# do 
#     echo "Running model: $model_name"
#     modified_model="${model_name//\//-}"
#     sbatch --job-name="$modified_model-scoring" --gpus=2 scripts/sbatch_action_scoring.sh "$model_name" 
# done

for model_name in "meta-llama/Llama-3.1-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct"
do 
    echo "Running model: $model_name"
    modified_model="${model_name//\//-}"
    sbatch --job-name="$modified_model-scoring" --gpus=4 scripts/sbatch_action_scoring.sh "$model_name" 
done