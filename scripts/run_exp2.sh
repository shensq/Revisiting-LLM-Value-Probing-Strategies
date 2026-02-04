#!/bin/sh

echo "Running experiment for demographic prompting" 
data_folder="data/demographic_prompting"

for model_name in "meta-llama/Llama-3.1-8B" "meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-7B-v0.3" "mistralai/Mistral-7B-Instruct-v0.3" "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-7B-Instruct" "tiiuae/falcon-7b-instruct"  "bigscience/bloomz-7b1" "Qwen/Qwen2.5-3B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
do 
    # for country in "China" "Egypt" "Mexico" "Czech" "Germany" "United States"
    for country in "China" "Egypt" "Mexico" "United States" "Germany"
    do
        # for variation in "baseline" "shuffle" "lower" "reverse" "num"
        for variation in "baseline" "reverse" "num"
        do 
            input_path="${data_folder}/${country}_${variation}.json"
            modified_model="${model_name//\//-}"
            modified_data="$(basename "$input_path" .json)"
            echo "Running input: ${input_path} on model: $modified_model"

            sbatch --job-name="$modified_model-$modified_data" scripts/sbatch.sh "$model_name" "$input_path"
        done
    done 
done

# ======== Model of different sizes ========

for model_name in "Qwen/Qwen2.5-14B-Instruct"
do 
    # for country in "China" "Czech" "Egypt" "Germany" "Mexico" "United States"
    for country in "China" "Egypt" "Mexico" "United States" "Germany"
    do
        # for variation in "baseline" "num" "shuffle" "lower" "reverse"
        for variation in "baseline" "reverse" "num"
        do 
            input_path="${data_folder}/${country}_${variation}.json"
            modified_model="${model_name//\//-}"
            modified_data="$(basename "$input_path" .json)"
            echo "Running input: ${input_path} on model: $modified_model"

            sbatch --job-name="$modified_model-$modified_data" --gpus=2 scripts/sbatch.sh "$model_name" "$input_path"
        done
    done 
done


for model_name in "meta-llama/Llama-3.1-70B-Instruct" "Qwen/Qwen2.5-72B-Instruct"
do 
    # for country in "China" "Czech" "Egypt" "Germany" "Mexico" "United States"
    for country in "China" "Egypt" "Mexico" "United States" "Germany"
    do
        # for variation in "baseline" "num" "shuffle" "lower" "reverse"
        for variation in "baseline" "reverse" "num"
        do 
            input_path="${data_folder}/${country}_${variation}.json"
            modified_model="${model_name//\//-}"
            modified_data="$(basename "$input_path" .json)"
            echo "Running input: ${input_path} on model: $modified_model"

            sbatch --job-name="$modified_model-$modified_data" --gpus=4 scripts/sbatch.sh "$model_name" "$input_path"
        done
    done 
done
