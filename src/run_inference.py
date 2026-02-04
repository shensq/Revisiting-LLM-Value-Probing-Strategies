"""
This module is used to run inference on the given processed data and model.

"""

import torch
import json
import argparse
import os
import copy

from credentials import access_token
from inference_utils import get_results
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="Get the inference results for questionare")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model to do inference on.")
    parser.add_argument("--input_path", type=str, default="data/baseline.json", help="The input dictionary containing all questions and options.")
    parser.add_argument("--num_return_sequences", type=int, default=10, help="The number of generated outputs per input.")
    parser.add_argument("--save_dir", type=str, default="results/", help="Folder path for saving the results.")
    parser.add_argument("--use_bf16", action="store_true", help="Whether to use fp16 for inference.")
    args = parser.parse_args()
    
    model_name_or_path = args.model_name_or_path
    data_name = args.input_path.split("/")[-1].split(".")[0]
    model_save_name = model_name_or_path.split("/")[-1]
    os.makedirs(os.path.join(args.save_dir, model_save_name), exist_ok=True)
    
    output_file_path = os.path.join(args.save_dir, model_save_name, f"{data_name}.json")
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists.")
        return 
    
    with open(args.input_path, mode="r") as f:
        input_dict = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token=access_token) # Give the access token if using LLAMA3.1 etc.
    
    if "Llama-3.1-70B-Instruct" in model_name_or_path or "Qwen2.5-72B-Instruct" in model_name_or_path:
        args.use_bf16 = True
        
    # Load 70B model with bfloat16
    if args.use_bf16:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=access_token, low_cpu_mem_usage=True, device_map="auto", torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=access_token, low_cpu_mem_usage=True, device_map="auto")
    
    # get the results dictionary given the model, input dictionary, and country
    results_dict = get_results(copy.deepcopy(input_dict), model, model_name_or_path, args.num_return_sequences, tokenizer, device)
    
    # with open(f"{args.save_dir}/{model_save_name}/{data_name}.json", mode="w") as f:
    with open(output_file_path, mode="w") as f:
        json.dump(results_dict, f, indent=4)
    return 
    
if __name__ == "__main__":
    main()