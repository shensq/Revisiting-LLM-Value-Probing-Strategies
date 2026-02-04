"""
This module is used specifically for scoring the action pairs to measure the value action agreement. 
"""
import json
import argparse
import os
import torch
from transformers import pipeline
from tqdm import tqdm 
from credentials import access_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get the scoring for action pairs")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="The model to do inference on.")
    parser.add_argument("--save_dir", type=str, default="results/", help="Folder path for saving the results.")
    parser.add_argument("--input_path", type=str, default="data/action_agreement/scoring_samples.jsonl", help="Path to the actions to score.")
    args = parser.parse_args()

    model_name_or_path = args.model_name_or_path
    model_save_name = model_name_or_path.split("/")[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_file_path = os.path.join(args.save_dir, model_save_name, "action_score.jsonl")
    if os.path.exists(output_file_path):
        print(f"File {output_file_path} already exists.")
        exit()
        
    with open(args.input_path, mode="r") as f:
        scoring_samples = [json.loads(line) for line in f]

    pipe = pipeline("text-generation", model=args.model_name_or_path, max_new_tokens=20, device_map="auto",token=access_token)
    
    for s in tqdm(scoring_samples[:], miniters=10):
        output = pipe(s['input'])
        output = output[0]['generated_text'][-1]['content']
        s['score'] = output
    
    os.makedirs(os.path.join(args.save_dir, model_save_name), exist_ok=True)
    
    with open(output_file_path, mode="w") as f:
        for s in scoring_samples:
            f.write(json.dumps(s) + "\n")
    
    