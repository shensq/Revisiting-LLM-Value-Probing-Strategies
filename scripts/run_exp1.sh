#!/bin/sh

echo "Running experiment for prompt variations and selection bias" 
# Run 8B sized models for different prompt and input variations. And 3B Qwen model
data_folder="data/base_variations"

for model_name in "meta-llama/Llama-3.1-8B" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-7B-Instruct" "tiiuae/falcon-7b-instruct"  "bigscience/bloomz-7b1" "Qwen/Qwen2.5-3B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
do 
    # echo "Running model: $model_name"
    # for input_path in "$data_folder/baseline.json" "$data_folder/num.json" "$data_folder/shuffle.json" "$data_folder/lower.json" "$data_folder/reverse.json"
    for input_path in "$data_folder/baseline.json" "$data_folder/num.json" "$data_folder/reverse.json"
    do
        # echo "Running input: $input_path"
        modified_model="${model_name//\//-}"
        modified_data="$(basename "$input_path" .json)"
        echo "$modified_model-$modified_data"
        sbatch --job-name="$modified_model-$modified_data" scripts/sbatch.sh "$model_name" "$input_path"
    done
done

# ======== Model of different sizes ========
# The same command, but requesting for a different number of GPUs for larger models 
for model_name in "Qwen/Qwen2.5-14B-Instruct" 
do 
    # echo "Running model: $model_name"
    # for input_path in "$data_folder/baseline.json" "$data_folder/num.json" "$data_folder/shuffle.json" "$data_folder/lower.json" "$data_folder/reverse.json"
    for input_path in "$data_folder/baseline.json" "$data_folder/num.json" "$data_folder/reverse.json"
    do
        # echo "Running input: $input_path"
        modified_model="${model_name//\//-}"
        modified_data="$(basename "$input_path" .json)"
        echo "$modified_model-$modified_data"
        sbatch --job-name="$modified_model-$modified_data" --gpus=2 scripts/sbatch.sh "$model_name" "$input_path"
    done
done


for model_name in "meta-llama/Llama-3.1-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct"
do 
    # echo "Running model: $model_name"
    # for input_path in "$data_folder/baseline.json" "$data_folder/num.json" "$data_folder/shuffle.json" "$data_folder/lower.json" "$data_folder/reverse.json"
    for input_path in "$data_folder/baseline.json" "$data_folder/num.json" "$data_folder/reverse.json"
    do
        # echo "Running input: $input_path"
        modified_model="${model_name//\//-}"
        modified_data="$(basename "$input_path" .json)"
        echo "$modified_model-$modified_data"
        sbatch --job-name="$modified_model-$modified_data" --gpus=4 scripts/sbatch.sh "$model_name" "$input_path"
    done
done