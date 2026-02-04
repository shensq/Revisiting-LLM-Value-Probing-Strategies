import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
from collections import Counter
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy,wasserstein_distance, pearsonr, spearmanr


def get_prob_distribution(sample,logits_column, seq_column, text_column, use_option_space):
    """Convert the inference results to probability distribution for each method. 
    The option is based on the logits of the answer choices, regardless of the label set or if it is normalized with option label prior. 
    The sequence is based on perplexity of the full answer.
    The text is based on parsed generation, whether from pattern matching or classification.
    """

    # Get all the probing results
    option_logits = sample[logits_column]
    seq_perplexity = sample[seq_column]
    generated_text = sample[text_column]

    # Convert them to probabilty distribution
    n = len(option_logits)
    if use_option_space:
        option_logits_space = sample["option_logits_space"]
        option_prob_both = softmax(option_logits + option_logits_space).tolist()
        option_prob = [option_prob_both[i] + option_prob_both[i+n] for i in range(n)]
    else:
        option_prob = softmax(option_logits).tolist()
        
    # Get the sequence likelihood and normalize, inverse of perplexity is proportional to likelihood
    seq_score = [1/s for s in seq_perplexity]
    seq_prob = [s/sum(seq_score) for s in seq_score]

    # Get the text answer distribution. Only count the valid answers after parsing. 
    valid_options = [s.split(".")[0] for s in sample['options']]
    counter = Counter(generated_text)
    total_valid_answers = sum([counter[o] for o in valid_options])
    text_prob = [counter[o]/len(generated_text) for o in valid_options]
    # The missing probability is evenly distributed to all the valid options
    text_compensate_prob = (1 - sum(text_prob))/len(valid_options)
    text_compensate_prob = max(0, text_compensate_prob) # make sure it is not negative
    text_prob = [p + text_compensate_prob for p in text_prob]
    
    # Group the options with the probability distribution, sorted by option label and then unpack
    # valid_options,option_prob, seq_prob, text_prob = zip(*sorted(zip(valid_options,option_prob, seq_prob, text_prob)))
    
    return valid_options, option_prob, seq_prob, text_prob, total_valid_answers

def get_all_prob(data,logits_column="option_logits", seq_column="full_answer_perplexities", text_column="parsed_generation", use_option_space=True):
    """Get the probability distribution for all the samples in the data, add to the loaded inference results as new columns.
    """
    for template in data:
        for question_id in data[template]:
            sample = data[template][question_id]
            valid_options, option_prob, seq_prob, text_prob,total_valid_answers = get_prob_distribution(sample, logits_column=logits_column, 
                                                                                    seq_column=seq_column, text_column=text_column, use_option_space=use_option_space)
            sample['valid_options'] = valid_options
            sample['option_probs'] = option_prob
            sample['seq_probs'] = seq_prob
            sample['text_probs'] = text_prob
            sample['total_valid_answers'] = total_valid_answers
    return data

def compute_mismatch_js(dist0, dist1):
    """
    Get the mismatch rate and JS distance for two distributions"""
    # Get accuracy and probability distance
    ## How much is the probing method affected by the prompt template? 
    # for each question and method, get the JS distance between different templates
    max_indices0 = [np.argmax(sublist) for sublist in dist0]
    max_indices1 = [np.argmax(sublist) for sublist in dist1]
    
    mismatch = sum([0 if i == j else 1 for i,j in zip(max_indices0, max_indices1)])/len(max_indices0)
    
    js_distance = [jensenshannon(dist0[i], dist1[i]) for i in range(len(dist0))]
    js_distance = [0 if np.isnan(x) else x for x in js_distance]
    js_distance = sum(js_distance)/len(js_distance)

    return mismatch, js_distance

def get_prompt_metrics(data):
    """
    Get the mismatch rate and JS distance between distributions from different templates with a set of inference. (model, variation)
    """
    template_method_probs = {}
    for template in data:
        option_probs = [data[template][question_id]["option_probs"] for question_id in data[template]]
        seq_probs = [data[template][question_id]["seq_probs"] for question_id in data[template]]
        text_probs = [data[template][question_id]["text_probs"] for question_id in data[template]]
        template_method_probs[template] = {"option_probs": option_probs, "seq_probs": seq_probs, "text_probs": text_probs}
    
    metrics = {}
    for method in ['option_probs', 'seq_probs', 'text_probs']:
        dists = [template_method_probs[t][method] for t in template_method_probs]
        mismatch = []
        js_distance = []
        
        for i in range(len(dists)):
            for j in range(i+1,len(dists)):
                # Get the mismatch rate and JS distance between distributions from different templates
                mismatch_ij, js_distance_ij = compute_mismatch_js(dists[i], dists[j])
                mismatch.append(mismatch_ij)
                js_distance.append(js_distance_ij)
        
        metrics[method] = {"mismatch": mismatch, "js_distance": js_distance}
        
    return metrics

def get_selection_metrics(all_results):
    """ Get the mismatch rate and JS distance between distributions from different label variations.
    The results for a variation comes from averaging over prompt templates. 
    all_results: {model: {variation: {method: data}}"""
    def average_template(probs):
        result = []
        # Iterate over columns (j)
        for j in range(len(probs[0])):  # Number of columns
            # Extract lists across all rows (i) for this column
            column_lists = [probs[i][j] for i in range(len(probs))]
            # Convert to NumPy array and compute mean along axis=0
            averaged_list = np.mean(np.array(column_lists), axis=0)
            result.append(list(averaged_list))
        return result
    
    metrics = {}
    # print(all_results.keys())
    for model in all_results:
        # Deal with each model
        variation_method_probs = {}
        # print(all_results[model].keys())
        for variation in all_results[model]:
            option_probs = []
            seq_probs = []
            text_probs = []
            for template in all_results[model][variation]:
                data = all_results[model][variation]
                option_probs.append([data[template][question_id]["option_probs"] for question_id in data[template]])
                seq_probs.append([data[template][question_id]["seq_probs"] for question_id in data[template]])
                text_probs.append([data[template][question_id]["text_probs"] for question_id in data[template]])
            
            # average over prompt templates
            
            option_probs = average_template(option_probs)
            seq_probs = average_template(seq_probs)
            text_probs = average_template(text_probs)
            
            variation_method_probs[variation] = {"option_probs": option_probs, "seq_probs": seq_probs, "text_probs": text_probs}

        metrics[model] = {}
        for method in ['option_probs', 'seq_probs', 'text_probs']:
            dists = [variation_method_probs[v][method] for v in variation_method_probs]
            mismatch = []
            js_distance = []
            # Get the mismatch rate and JS distance between distributions from different variations
            for i in range(len(dists)):
                for j in range(i+1,len(dists)):
                    # Get the mismatch rate and JS distance between distributions from different templates
                    mismatch_ij, js_distance_ij = compute_mismatch_js(dists[i], dists[j])
                    mismatch.append(mismatch_ij)
                    js_distance.append(js_distance_ij)
            metrics[model][method] = {"mismatch": mismatch, "js_distance": js_distance}
    return metrics

def draw_robustness_old(results, save_name=None):
    # Extracting data for plotting
    models = results.keys()

    option_metrics = [value['option_probs'] for value in results.values()]
    sequence_metrics = [value['seq_probs'] for value in results.values()]
    text_metrics = [value['text_probs'] for value in results.values()]

    option_mismatch = [x['mismatch'] for x in option_metrics]
    sequence_mismatch = [x['mismatch'] for x in sequence_metrics]
    text_mismatch = [x['mismatch'] for x in text_metrics]
    option_js = [x['js_distance'] for x in option_metrics]
    sequence_js = [x['js_distance'] for x in sequence_metrics]
    text_js = [x['js_distance'] for x in text_metrics]
    
    avg_mismatch_data = [option_mismatch, sequence_mismatch, text_mismatch]
    avg_js_data = [option_js, sequence_js, text_js]


    # Plotting
    x = np.arange(len(models))
    width = 0.2
    plt.rc('font', size=18) 
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    methods = ["Token", "Seq", "Text"]
    
    # Plot Average Mismatch
    for i, method in enumerate(methods):
        ax1.bar(x + i * width, avg_mismatch_data[i], width, label=method)
    # ax1.set_title("Comparison of Average Mismatch by Method")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(models, rotation=25, ha="right")
    ax1.set_ylabel("Average Mismatch")
    # ax1.legend(title="Method")

    # Plot Average JS Distance
    for i, method in enumerate(methods):
        ax2.bar(x + i * width, avg_js_data[i], width, label=method)
    # ax2.set_title("Comparison of Average JS Distance by Method")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(models, rotation=25, ha="right")
    ax2.set_ylabel("Average JS Distance")
    ax2.legend(title="Method", loc="upper right",bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    plt.savefig(f"./writeup/{save_name}.pdf", dpi=300, bbox_inches='tight')
    return

def draw_robustness(results, save_name=None):
    def convert_to_df(results, metric=None):
        data = {
            model: {
                'Token': details['option_probs'][metric],
                'Seq': details['seq_probs'][metric],
                'Text': details['text_probs'][metric],
            }
            for model, details in results.items()
        }

        # Convert data to DataFrame for easier plotting
        df = pd.DataFrame(data).T
        df = df[['Token', 'Seq', 'Text']]  # Order columns
        df.index.name = 'Model'

        # Order models by size based on naming convention
        ordered_models = sorted(df.index, key=lambda x: float(x.split('-')[1].replace('B', '').replace('I', '')))
        df = df.loc[ordered_models]

        # Update model names to merge instruction-tuned and non-instruction models
        df['Model Type'] = ['Instruction' if 'I' in model else 'Non-Instruction' for model in df.index]
        df['Base Model'] = [model.replace('-I', '') for model in df.index]

        # Sort models by size (excluding "I")
        # ordered_base_models = sorted(df['Base Model'].unique(), key=lambda x: float(x.split('-')[1].replace('B', '')))
        
        ordered_base_models = ['Qwen2.5-3B',
                        'Llama3.2-3B',
                        'Bloomz-7B',
                        'Falcon-7B',
                        'Mistral-7B',
                        'Qwen2.5-7B',
                        'Llama3.1-8B',
                        'Qwen2.5-14B',
                        'Llama3.1-70B',
                        'Qwen2.5-72B'] # Set the order manually
        df['Base Model'] = pd.Categorical(df['Base Model'], categories=ordered_base_models, ordered=True)

        # Reset index to prepare for seaborn
        df.reset_index(drop=True, inplace=True)

        # Melt for seaborn
        df_melted = df.melt(id_vars=['Base Model', 'Model Type'], var_name='Method', value_name=metric)
        return df_melted

    sns.set_style("whitegrid")
    # fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharey=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharey=False)
    df_mismatch = convert_to_df(results, metric='mismatch')
    sns.lineplot(
        data=df_mismatch,
        x='Base Model',
        y='mismatch',
        hue='Method',
        style='Model Type',
        markers=True,
        dashes=True,
        markersize=10,
        ax=axes[0]
    )
    axes[0].set_xlabel('', fontsize=16)
    axes[0].set_ylabel('Mismatch', fontsize=16)
    axes[0].tick_params(axis='x', rotation=15, labelsize=14)  # For x-axis ticks
    axes[0].tick_params(axis='y', labelsize=14)              # For y-axis ticks
    axes[0].legend(loc='best')      # Set legend location

    df_js = convert_to_df(results, metric='js_distance')
    sns.lineplot(
        data=df_js,
        x='Base Model',
        y='js_distance',
        hue='Method',
        style='Model Type',
        markers=True,
        dashes=True,
        markersize=10,
        ax=axes[1]
    )
    axes[1].set_xlabel('', fontsize=16)
    axes[1].set_ylabel('JS Distance', fontsize=16)
    axes[1].tick_params(axis='x', rotation=15, labelsize=14)  # For x-axis ticks
    axes[1].tick_params(axis='y', labelsize=14)              # For y-axis ticks
    axes[1].legend(loc='best').set_visible(False)                      # Set legend location

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"./writeup/{save_name}.pdf", dpi=300, bbox_inches='tight')
    return


# For Exp2 demogrpahic prompting 

def process_human_results(human_results):
    """Process the human results to make the option index consistent with the model results"""
    # Load only the questions used in the experiment
    with open('./data/input_dict.json', 'r') as file:
        input_dict = json.load(file)
    input_dict = input_dict['template0']
    probing_qids = list(input_dict.keys())
    
    # These quesions' option index on the questionnaire starts from 0 
    qid_zero_start = ["Q"+str(i) for i in range(94,106)] + ["Q119"] + ["Q"+str(i) for i in range(122,130)]

    for country in human_results:
        for qid in human_results[country]:
            try:
                if qid not in probing_qids:
                    continue
                dist = human_results[country][qid]
                # Shift the index to 0 start
                if qid not in qid_zero_start:
                    dist = {str(int(k)-1): v for k,v in dist.items()}
                # keep only keys from 0-10
                dist = {k: dist[k] for k in dist if k in [str(i) for i in range(0,10)]}
                # convert to list of float 
                labels = sorted([int(i) for i in dist.keys()])
                num_options = len(input_dict[qid]['options'])
                probs = [0] * num_options
                human_response_count = sum(dist.values())
                if human_response_count != 0:
                    for i in labels:
                        probs[i] = dist[str(i)]/human_response_count
                    
                human_results[country][qid] = probs
            except Exception as e:
                # Some questions are not answered by some countries. Not in the used set of questions. 
                print(e, country, qid, " Check human results")
                continue
    return human_results   

def get_human_results():
    """Load the human results from the WVS survey
    
    Returns:
    human_results: dict. Accessed by human_results[country][question_id] = {'1': 1159, '2': 88, '3': 7, '4': 3}
    """
    human_results = {}
    with open('./data/wvs_survey_by_country.json','r') as f:
        human_results = json.load(f)
    human_results = process_human_results(human_results)
    human_results['United States'] = human_results["United States of America"]
    return human_results

def get_average_EMD(human_dist, model_dists):
    def get_EMD(h_d, m_d):
        """Calculate the Earth Mover's Distance between a set of human results and a set of model results"""
        "The number of questions should be the same"
        if len(h_d) == len(m_d) and sum(h_d) != 0 and sum(m_d) != 0:
            weights1 = np.array(h_d)
            weights2 = np.array(m_d)
            emd = wasserstein_distance(range(len(weights1)), range(len(weights2)), weights1, weights2)
            return emd
        else:
            return None
        
    def get_JS(h_d, m_d):
        if len(h_d) == len(m_d) and sum(h_d) != 0 and sum(m_d) != 0:
            js = jensenshannon(h_d, m_d)
            return js
        else:
            return None
        
    average_emd = {"option_probs":0, "seq_probs":0, "text_probs":0}
    for template in model_dists.keys():
        question_ids = set(human_dist.keys()).intersection(set(model_dists[template].keys()))
        emds = {"option_probs":{}, "seq_probs":{}, "text_probs":{}}

        for scoring_method in emds.keys():
            for qid in question_ids:
                h_d = human_dist[qid]    
                m_d = model_dists[template][qid][scoring_method]
                emds[scoring_method][qid] = get_EMD(h_d, m_d)
                # emds[scoring_method][qid] = get_JS(h_d, m_d)
            # Do the average over valid results
            average_emd[scoring_method] += np.mean([v for v in emds[scoring_method].values() if v is not None])
    for scoring_method in emds.keys():
        average_emd[scoring_method] /= len(model_dists)

    return average_emd

def get_delta_alignment(country_alignment, wo_country_alignemnt):
    # delta_alignment = {"absolute":{}, "relative":{}}
    # for scoring_method in country_alignment.keys():
    #     delta_alignment['absolute'][scoring_method] = wo_country_alignemnt[scoring_method] - country_alignment[scoring_method]
    #     delta_alignment['relative'][scoring_method] = (wo_country_alignemnt[scoring_method] - country_alignment[scoring_method]) / wo_country_alignemnt[scoring_method]
    # return delta_alignment
    return {"country": country_alignment, "wo_country": wo_country_alignemnt}
    
def draw_demographic_alignment(results):
    records = []
    for (model, variation, region), scores in results.items():
        for score_type, values in scores.items():
            record = {'Model': model,'Variation': variation,'Region': region, 'Type': score_type, **values}
            records.append(record)

    df = pd.DataFrame(records)

    df = df.rename(columns={
        "option_probs": "Token",
        "seq_probs": "Seq",
        "text_probs": "Text"
    })
    
    df = df.groupby(["Model", "Type"]).mean(numeric_only=True).reset_index()
    
    ordered_base_models = ['Qwen2.5-3B',
                        'Llama3.2-3B',
                        'Bloomz-7B',
                        'Falcon-7B',
                        'Mistral-7B',
                        'Qwen2.5-7B',
                        'Llama3.1-8B',
                        'Qwen2.5-14B',
                        'Llama3.1-70B',
                        'Qwen2.5-72B'] # Set the order manually
    
    
    df = df[df["Model"].str.contains("-I")]

    # Remove '-I' from the model column for comparison
    df["model_base"] = df["Model"].str.replace("-I", "", regex=False)
    # Add a sort key based on the order in ordered_base_models
    df["sort_key"] = df["model_base"].apply(lambda x: ordered_base_models.index(x) if x in ordered_base_models else float('inf'))
    # Sort the DataFrame by the sort key
    df = df.sort_values("sort_key").drop(columns=["sort_key", "model_base"])

    # Calculate differences
    country_data = df[df["Type"] == "country"].set_index("Model")
    wo_country_data = df[df["Type"] == "wo_country"].set_index("Model")

    # Difference DataFrame
    diff_data = wo_country_data[["Token", "Seq", "Text"]] - country_data[["Token", "Seq", "Text"]]
    diff_data.reset_index(inplace=True)
    models = diff_data["Model"].unique()
    x = np.arange(len(models))
    width = 0.2

    # Plot settings
    plt.rc('font', size=16)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))


    # Bar chart for differences
    ax.bar(x - width, diff_data["Token"], width, label="Token")
    ax.bar(x, diff_data["Seq"], width, label="Seq")
    ax.bar(x + width, diff_data["Text"], width, label="Text")

    # Labels and Title
    # ax.set_title(r"Change from Country to Without Country ($\Delta$)")
    # ax.set_xticks(x)
    xtick_positions = x + 0.5
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(models, rotation=15, ha="right")


    ax.set_ylabel("Alignment improvement in EMD")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Add a reference line at 0
    ax.legend()
    # ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, fontsize=14, frameon=False)


    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig("./writeup/demograpic_prompting.pdf", dpi=300, bbox_inches='tight')
    return 
    
    # # Separate data by Type
    # absolute_data = df[df["Type"] == "absolute"]
    # relative_data = df[df["Type"] == "relative"]

    # # Define models and positions for bars
    # models = absolute_data["Model"].unique()
    # x = np.arange(len(models))
    # width = 0.2
    # plt.rc('font', size=18) 
    # plt.style.use('seaborn-v0_8-whitegrid')
    # # Create the plot
    # fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=True)

    # # Subplot 1: Absolute
    # axes[0].bar(x - width, absolute_data["option"], width, label="option")
    # axes[0].bar(x, absolute_data["seq"], width, label="seq")
    # axes[0].bar(x + width, absolute_data["text"], width, label="text")
    # axes[0].set_title(r"Absolute Improvement ($\uparrow$)")
    # axes[0].set_xticks(x)
    # axes[0].set_xticklabels(models, rotation=25, ha="right")
    # axes[0].set_ylabel(r"$\Delta$ EMD with demographic information")
    # # axes[0].legend()

    # # Subplot 2: Relative
    # axes[1].bar(x - width, relative_data["option"], width, label="option")
    # axes[1].bar(x, relative_data["seq"], width, label="seq")
    # axes[1].bar(x + width, relative_data["text"], width, label="text")
    # axes[1].set_title(r"Relative Improvement ($\uparrow$)")
    # axes[1].set_xticks(x)
    # axes[1].set_xticklabels(models, rotation=25, ha="right")
    # axes[1].legend()

    # # Adjust layout
    # plt.tight_layout()
    # # plt.show()
    # plt.savefig("./writeup/demograpic_prompting.pdf", dpi=300, bbox_inches='tight')
    # return 



# For Exp3 action agreement
def load_action_scores(model_name, results_dir="./results"):
    # Load the JSONL file
    with open(f"{results_dir}/{model_name}/action_score.jsonl", 'r') as file:
        data = file.readlines()

    # Parse the data
    parsed_data = [json.loads(line) for line in data]

    # Transform into a DataFrame
    records = []
    pattern = "\d+"

    for entry in parsed_data:
        score = entry["score"]
        score = re.findall(pattern, score)
        if not score:
            score = -1
        else:
            score = int(score[0])
        
        record = {
            "qid": entry["qid"],
            "action": entry["action"],
            # "question": entry["question"],
            "content": entry["input"][0]["content"],
            "score": score  # Clean up the score value
        }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.groupby(["qid", "action"], as_index=False)["score"].mean()
    action_scores = df.groupby("qid").apply(lambda group: group.set_index("action")["score"].to_dict()).to_dict()
    return action_scores

def get_action_score_correlation(action_scores, model_probs):
    def aggregate_model_probs(model_probs):
        prob_distributions = [] 
        for variant in model_probs:
            for template in model_probs[variant]:
                prob_distributions.append((variant, template))

        qids = model_probs['baseline']['template0'].keys()
        aggregated_probs = {}
        for qid in qids:
            aggregated_probs[qid] = {}
            for scoring_method in ["option_probs", "seq_probs", "text_probs"]:
                probs_all = []
                for v,t in prob_distributions:
                    probs_all.append(model_probs[v][t][qid][scoring_method])
                aggregated_probs[qid][scoring_method] = list(np.mean(probs_all, axis=0))
        return aggregated_probs
        
    model_probs = aggregate_model_probs(model_probs)
    dict_method_scores_prob = {}
    correlation_results = []
    
    for scoring_method in ["option_probs", "seq_probs", "text_probs"]:
        dict_method_scores_prob[scoring_method] = {}
        dict_method_scores_prob[scoring_method]['action_score'] = []
        dict_method_scores_prob[scoring_method]['value_dist'] = []

        question_ids = set(model_probs.keys()).intersection(action_scores.keys())
        for qid in question_ids:
            c_q_dist = model_probs[qid][scoring_method]
            action_score = action_scores[qid]
            dict_method_scores_prob[scoring_method]['value_dist'].extend(c_q_dist)
            dict_method_scores_prob[scoring_method]['action_score'].extend([action_score['pos'], action_score['neg']])

        # Calculate the correlation
        var_option = dict_method_scores_prob[scoring_method]['value_dist']
        var_action = dict_method_scores_prob[scoring_method]['action_score']
        
        cor_pearson, p_value = pearsonr(var_option, var_action)
        correlation_results.append({"scoring_method": scoring_method, "correlation": cor_pearson, "p_value": p_value, "metric": "pearson"})
        # print(f"{scoring_method} Pearson correlation: {cor_pearson} (p-value: {p_value})")
        
        cor_spearman, p_value = spearmanr(var_option, var_action)
        correlation_results.append({"scoring_method": scoring_method, "correlation": cor_spearman, "p_value": p_value, "metric": "spearman"})
        # print(f"{scoring_method} Spearman correlation: {cor_spearman} (p-value: {p_value})")
    
    return correlation_results

def draw_action_correlation(correlation_results):
    records = []
    for model, stats in correlation_results.items():
        for entry in stats:
            record = {'model': model, **entry}
            records.append(record)

    ordered_base_models = ['Qwen2.5-3B',
                    'Llama3.2-3B',
                    'Bloomz-7B',
                    'Falcon-7B',
                    'Mistral-7B',
                    'Qwen2.5-7B',
                    'Llama3.1-8B',
                    'Qwen2.5-14B',
                    'Llama3.1-70B',
                    'Qwen2.5-72B'] # Set the order manually
    df = pd.DataFrame(records)
    scoring_method_mapping = {
    "option_probs": "Token",
    "seq_probs": "Seq",
    "text_probs": "Text"
    }

    # Replace values in the 'scoring_method' column
    df["scoring_method"] = df["scoring_method"].replace(scoring_method_mapping)
    # Remove '-I' from the model column for comparison
    df["model_base"] = df["model"].str.replace("-I", "", regex=False)
    # Add a sort key based on the order in ordered_base_models
    df["sort_key"] = df["model_base"].apply(lambda x: ordered_base_models.index(x) if x in ordered_base_models else float('inf'))
    # Sort the DataFrame by the sort key
    df = df.sort_values("sort_key").drop(columns=["sort_key", "model_base"])

    # Reset index
    df = df.reset_index(drop=True)


    # Filter data by metric
    pearson_data = df[df["metric"] == "pearson"]
    spearman_data = df[df["metric"] == "spearman"]


    plt.style.use('seaborn-v0_8-whitegrid')
    # Get unique models and scoring methods
    models = df["model"].unique()
    methods = ["Token", "Seq", "Text"]

    x = np.arange(len(models))  # X positions for models
    width = 0.2  # Width of each bar

    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # Subplot 1: Pearson
    for i, method in enumerate(methods):
        method_data = pearson_data[pearson_data["scoring_method"] == method]
        bars = axes[0].bar(
            x + (i - 1) * width,
            method_data["correlation"],
            width,
            label=method
        )
        # Add markers for p-value > 0.05
        for bar, p_value in zip(bars, method_data["p_value"]):
            if p_value > 0.05:
                axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '*', ha='center', va='bottom')

    axes[0].set_title("Pearson Correlation")
    # axes[0].set_xticks(x)
    xtick_positions = x + 0.5
    axes[0].set_xticks(xtick_positions)
    axes[0].set_xticklabels(models, rotation=15, ha="right")
    axes[0].set_ylabel("Correlation")
    axes[0].legend()
    # axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, fontsize=10, frameon=False)
    # Subplot 2: Spearman
    for i, method in enumerate(methods):
        method_data = spearman_data[spearman_data["scoring_method"] == method]
        bars = axes[1].bar(
            x + (i - 1) * width,
            method_data["correlation"],
            width,
            label=method
        )
        # Add markers for p-value > 0.05
        for bar, p_value in zip(bars, method_data["p_value"]):
            if p_value > 0.05:
                axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), '*', ha='center', va='bottom')

    axes[1].set_title("Spearman Correlation")
    # axes[1].set_xticks(x)
    xtick_positions = x + 0.5
    axes[1].set_xticks(xtick_positions)
    axes[1].set_xticklabels(models, rotation=15, ha="right")
    axes[1].legend()
    # axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1), ncol=3, fontsize=10, frameon=False)
    # Adjust layout
    plt.tight_layout()
    # plt.show()
    plt.savefig("./writeup/action_agreement.pdf", dpi=300, bbox_inches='tight')
    return    
