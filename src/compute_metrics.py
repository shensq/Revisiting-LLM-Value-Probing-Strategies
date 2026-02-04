"""
This module takes the generated inferences and computes the metrics for the probing methods.
"""
import os
import json
import argparse
from parse_generation import parse_geneartion_pattern
from eval_utils import * 

def evaluate_robustness_prompt(models, results_files, result_folder, args, name_mapping=None):
    # Get the results for robustness on prompt. Each (model, method) comparing N different prompts
    
    metrics_robustness_prompt = {}
    for model in models:
        for source in ['baseline.json']:
            # Find results for this model and source
            file = [f for f in results_files if f"{model}/{source}" in f][0]
            with open(result_folder+file, 'r') as f:
                data = json.load(f)
        
            data = get_all_prob(data,logits_column=args.logits_column, seq_column=args.seq_column, 
                                text_column=args.text_column, use_option_space=args.use_option_space)
            
            metrics = get_prompt_metrics(data)
            
            metrics_robustness_prompt[model] = metrics
            
    # Save the results in case
    # metrics_robustness_prompt: {model: {method: {mismatch: [mismatch_01, ...], js_distance: [js_distance_01, ...]}}}
    with open("./writeup/metrics_robustness_prompt.json", "w") as f:
        json.dump(metrics_robustness_prompt, f, indent=4)
    
    # Calculate the average mismatch and JS distance for each model and method
    for model in models:
        for method in ['option_probs', 'seq_probs', 'text_probs']:
            metrics_robustness_prompt[model][method]["mismatch"] = sum(metrics_robustness_prompt[model][method]["mismatch"])/len(metrics_robustness_prompt[model][method]["mismatch"])
            # metrics_robustness_prompt[model][method]["mismatch"] = round(metrics_robustness_prompt[model][method]["mismatch"], 3)
            metrics_robustness_prompt[model][method]["js_distance"] = sum(metrics_robustness_prompt[model][method]["js_distance"])/len(metrics_robustness_prompt[model][method]["js_distance"])
            # metrics_robustness_prompt[model][method]["js_distance"] = round(metrics_robustness_prompt[model][method]["js_distance"], 3)
    
    def rename(key):
        if key in name_mapping:
            return name_mapping[key]
        else:
            return key
    metrics_robustness_prompt = {rename(key): value for key, value in metrics_robustness_prompt.items()}

    draw_robustness(metrics_robustness_prompt, save_name="robustness_prompt")
    return

def save_valid_ratio(models, variations, all_results):
    f = open("writeup/valid_ratio.csv",'w')
    for model in models:
        for variation in variations:
            for template in all_results[model][variation]:
                average_valid = sum(all_results[model][variation][template][x]['total_valid_answers'] for x in all_results[model][variation][template])/len(all_results[model]['baseline']['template0'])
                f.write(f"{model},{variation},{template},{average_valid}\n")
    f.close()
    return

def evaluate_robustness_selection(models, results_files, result_folder, args, name_mapping=None):
    # Get the results for robustness on selection bias. Each (model, method) has N different prompts and M different label variations

    # variations = ["baseline", "shuffle", "lower", "reverse", "num"]
    variations = ["baseline", "reverse", "num"]
    # variations = ["baseline", "reverse"]
    
    all_results = {}
    for model in models:
        all_results[model] = {}
        for variation in variations:
            # Find results for this model and source
            try:
                file = [f for f in results_files if f"{model}/{variation}.json" in f][0]
                with open(result_folder+file, 'r') as f:
                    data = json.load(f)
                data = get_all_prob(data,logits_column=args.logits_column, seq_column=args.seq_column, 
                                text_column=args.text_column, use_option_space=args.use_option_space)
                all_results[model][variation] = data
            except:
                print(f"{model}/{variation}.json not found.")

    save_valid_ratio(models, variations, all_results)
    metrics_robustness_selection = get_selection_metrics(all_results)
            
    # Save the results in case
    # metrics_robustness_selection: {model: {variation: {method: {mismatch: [mismatch_01, ...], js_distance: [js_distance_01, ...]}}}}
    with open("./writeup/metrics_robustness_selection.json", "w") as f:
        json.dump(metrics_robustness_selection, f, indent=4)
    
    # Calculate the average mismatch and JS distance for each model and method
    for model in models:
        for method in ['option_probs', 'seq_probs', 'text_probs']:
            metrics_robustness_selection[model][method]["mismatch"] = sum(metrics_robustness_selection[model][method]["mismatch"])/len(metrics_robustness_selection[model][method]["mismatch"])
            # metrics_robustness_prompt[model][method]["mismatch"] = round(metrics_robustness_prompt[model][method]["mismatch"], 3)
            metrics_robustness_selection[model][method]["js_distance"] = sum(metrics_robustness_selection[model][method]["js_distance"])/len(metrics_robustness_selection[model][method]["js_distance"])
            # metrics_robustness_prompt[model][method]["js_distance"] = round(metrics_robustness_prompt[model][method]["js_distance"], 3)
    
    def rename(key):
        if key in name_mapping:
            return name_mapping[key]
        else:
            return key
    metrics_robustness_selection = {rename(key): value for key, value in metrics_robustness_selection.items()}

    draw_robustness(metrics_robustness_selection, save_name="robustness_selection")
    return

def evaluate_demograpic_prompting(models, name_mapping=None):
    countries = ["China", "Egypt", "Mexico","United States","Germany"]
    human_results =get_human_results()
    
    results = {}
    question_options = pd.read_csv('./data/question_options.csv')

    # csv_fp = open("./writeup/demographic_alignment.csv", 'w')
    # csv_fp.write("model,variation,country,alignment\n")
    
    # Find the rows where the column Ordinal is False
    non_ordinal_QID = question_options[question_options['Ordinal'] == False]['ID']
    def remove_non_ordinal(data):
        for template in data:
            for qid in non_ordinal_QID:
                if qid in data[template]:
                    del data[template][qid]
        return data
    
    for model in models:
    # for model in ["Llama-3.1-8B-Instruct"]:
        for variation in ["baseline", "reverse", "num"]:
        # for variation in ["baseline"]:
            # print(f"Reading {model} results on {variation}")
            try:
                with open(f"./results/{model}/{variation}.json",'r') as f:
                    base_model_results = json.load(f)
                    model_probs_wodp = get_all_prob(base_model_results)
                    model_probs_wodp = remove_non_ordinal(model_probs_wodp)
                model_probs_dp = {}
                for country in countries:
                    # print(f"Reading {model} results on {country}")
                    # parse_geneartion_pattern(f"./results/{model}/{country}_{variation}.json") 
                    with open(f"./results/{model}/{country}_{variation}.json",'r') as f:
                    # parse_geneartion_pattern(f"./results/{model}/{country}.json") 
                    # with open(f"./results/{model}/{country}.json",'r') as f:
                        dp_model_results = json.load(f)
                    model_probs_dp[country] = get_all_prob(dp_model_results)
                    model_probs_dp[country] = remove_non_ordinal(model_probs_dp[country])
                    human_dist = human_results[country]
                    wodp_alignemnt = get_average_EMD(human_dist, model_probs_wodp)
                    country_alignment = get_average_EMD(human_dist, model_probs_dp[country])
                    delta_alignment = get_delta_alignment(country_alignment, wodp_alignemnt)
                    # print(f"{model} on {country}: {delta_alignment}")
                    # write_fp.write(json.dumps({model: {country: delta_alignment}}) + "\n")
                    # csv_fp.write(f"{name_mapping[model]},{variation},{country},{delta_alignment}\n")
                    results[(model,variation,country)] = delta_alignment
            except Exception as e:
                print(e)
                print(f"Results on {model} {variation} with demographic prompting not found")
                
    def rename(key):
        model, variation, country = key
        if model in name_mapping:
            return (name_mapping[model], variation, country)
        else:
            return key
    results = {rename(key): value for key, value in results.items()}
   
    with open("writeup/alignment.jsonl", 'w') as f:
        for item in results.items():
            f.write(json.dumps(item) + '\n')

    draw_demographic_alignment(results)
    return 

def evaluate_action_correlation(models, results_files, result_folder, args, name_mapping=None):
    """Evaluate the correlation between the action score and the probability distribution of the model"""
    models = [m for m in models if "instruct" in m.lower()]
    variations = ["baseline", "reverse", "num"]
    model_probs = {}
    for model in models:
        model_probs[model] = {}
        for variation in variations:
            # Find results for this model and source
            try:
                file = [f for f in results_files if f"{model}/{variation}.json" in f][0]
                with open(result_folder+file, 'r') as f:
                    data = json.load(f)
                data = get_all_prob(data,logits_column=args.logits_column, seq_column=args.seq_column, 
                                text_column=args.text_column, use_option_space=args.use_option_space)
                model_probs[model][variation] = data
            except:
                print(f"{model}/{variation}.json not found.")

    
    # Group model prob into two bins 
    for model in model_probs:
        for variation in variations:
            try:
                for template in model_probs[model][variation]:
                    for qid in model_probs[model][variation][template]:
                        for scoring_method in ["option_probs", "seq_probs", "text_probs"]:
                            c_q_dist = model_probs[model][variation][template][qid][scoring_method]
                            num_options = len(c_q_dist)
                            model_probs[model][variation][template][qid][scoring_method] = [sum(c_q_dist[:num_options//2]), sum(c_q_dist[num_options//2:])]
            except:
                print(f"Error in getting model prob for {model} {variation}")
    
    # Load the action scores
    action_scores = {}
    for model in list(model_probs.keys()):
        try:
            action_scores[model] = load_action_scores(model)
        except:
            print(f"Error in getting action score for {model}")
            del model_probs[model] # Not calculating correlation for this model
            
    correlation_results = {}
    for model in action_scores:
        correlation_results[model] = get_action_score_correlation(action_scores[model], model_probs[model])

    def rename(key):
        if key in name_mapping:
            return name_mapping[key]
        else:
            return key
    correlation_results = {rename(key): value for key, value in correlation_results.items()}

    with open("./writeup/correlation_results.json", "w") as f:
        json.dump(correlation_results, f, indent=4)
    
    draw_action_correlation(correlation_results)
    return 
    
def main():
    parser = argparse.ArgumentParser(description="Get the metrics based on the inference results.")
    parser.add_argument("--logits_column", type=str, default="option_logits", help="The column name for the logits of the answer choices.")
    parser.add_argument("--use_option_space", type=bool, default=True, help="Whether to use the option space for the probability distribution.")
    parser.add_argument("--seq_column", type=str, default="full_answer_perplexities", help="The column name for the perplexity of the full answer.")
    parser.add_argument("--text_column", type=str, default="parsed_generation", help="The column name for the parsed generation.")
    parser.add_argument("--force_reparsing", type=bool, default=False, help="Whether to force reparsing the generation.")
    args = parser.parse_args()
    
    # Parse the generation if not already done
    if args.force_reparsing:
        parse_geneartion_pattern()
        
    # Get all the inference results
    result_folder = "./results/"
    results_files = []
    for root, dirs, files in os.walk(result_folder):
        for file in files:
            results_files.append(os.path.join(root, file))
    results_files = [f.replace(result_folder,"") for f in results_files] # Check for existing results
    # # Use all the models and data sources
    # models = list(set([f.split('/')[0] for f in results_files]))
    # data_source = list(set([f.split('/')[1] for f in results_files]))

    name_mapping= {'Mistral-7B-Instruct-v0.3': 'Mistral-7B-I', 
                   'Mistral-7B-v0.3': 'Mistral-7B',
                   'Qwen2.5-7B': 'Qwen2.5-7B',
                   'Qwen2.5-7B-Instruct': 'Qwen2.5-7B-I',
                   'Llama-3.1-8B': 'Llama3.1-8B',
                   'Llama-3.1-8B-Instruct': 'Llama3.1-8B-I',
                   'bloomz-7b1': 'Bloomz-7B-I',
                   'falcon-7b-instruct': 'Falcon-7B-I',
                   "Qwen2.5-3B-Instruct":"Qwen2.5-3B-I",
                   "Llama-3.2-3B-Instruct":"Llama3.2-3B-I",
                   "Qwen2.5-14B-Instruct":"Qwen2.5-14B-I",
                   "Llama-3.1-70B-Instruct": "Llama3.1-70B-I",
                   "Qwen2.5-72B-Instruct":"Qwen2.5-72B-I",
                   }

    # Specify the models and data sources
    models = list(name_mapping.keys())
    # models = ['Llama-3.1-8B','Llama-3.1-8B-Instruct', 'Mistral-7B-v0.3', 'Mistral-7B-Instruct-v0.3', 'Qwen2.5-7B', 'Qwen2.5-7B-Instruct','bloomz-7b1', 'falcon-7b-instruct']
    print(models)
    
    # evaluate_robustness_prompt(models, results_files, result_folder, args, name_mapping)
    # evaluate_robustness_selection(models, results_files, result_folder, args, name_mapping)
    # evaluate_demograpic_prompting(models, name_mapping)
    evaluate_action_correlation(models, results_files, result_folder, args, name_mapping)
    
    return
    
if __name__ == "__main__":
    main()
