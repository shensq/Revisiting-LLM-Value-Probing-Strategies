"""
This module construct all the required input for different set of experiments.
"""
import pandas as pd
import os
import json
import copy
import random 
from unidecode import unidecode


def create_input_json(question_option_df, NUM_TEMPLATES, TEMPLATE_DIR, OUTPUT_FP, OPTION_LABELS,option_labels_map,is_reversed=False):
    """Create the input json file based on the templates and the question options
    """
    input_dict = dict()

    for template_idx in range(NUM_TEMPLATES):
        template_name = f"template{template_idx}"
        template_fp = os.path.join(TEMPLATE_DIR, f"{template_name}.json")

        with open(template_fp, mode="r") as f:
            template = json.load(f)
            
        input_dict[template_name] = dict()

        for i in range(len(question_option_df)):
            row = question_option_df.iloc[i]
            question_id = row["ID"]
            question = row["Question"]
            # Adding a patch for the reversed order
            if not is_reversed:
                options = [f"{option_labels_map[c]}. {row[c]}" for c in OPTION_LABELS if not pd.isnull(row[c])]
            else:
                o_l = [c for c in OPTION_LABELS if not pd.isnull(row[c])] # Get the number of options for this question
                o_l_mapping = dict(zip(o_l, reversed(o_l))) # Create a mapping from the original label to the reversed label
                options = [f"{o_l_mapping[c]}. {row[c]}" for c in o_l]
                
            options_str = "\n".join(options)

            inp = copy.deepcopy(template)
            for j in range(len(inp)):
                inp[j]["content"] = inp[j]["content"].replace("[question]", question)
                inp[j]["content"] = inp[j]["content"].replace("[options]", options_str)
                inp[j]["content"] = unidecode(inp[j]["content"])
            
            input_dict[template_name][question_id] = {
                "input": inp,
                "options": options
            }
            
    with open(OUTPUT_FP, mode="w") as f:
        json.dump(input_dict, f, indent=4)

def get_action_agreement_input():
    situation_actions = []
    data_folder = "data/action_agreement"
    with open(f"{data_folder}/value_tasks_verified_parsed.json", 'r', encoding='UTF-8') as f:
        for line in f:
            situation_actions.append(json.loads(line))
    question_options_df = pd.read_csv('data/question_options.csv')

    # Build the dictionary
    question_options_dict = question_options_df.set_index('Question')['ID'].to_dict()

    situation_actions_filtered = []
    for s in situation_actions:
        try: 
            verification = json.loads(s["Verification"])
            if verification['Q1'] == "Yes" and verification['Q2'] == "Yes" and verification['Q3'] == "Yes" and verification['Q4'] == "Yes":
                situation_actions_filtered.append(s)
                
        except:
            continue
        
    scoring_samples = []
    sample_counter = {}
    template1="""Imagine the following situation: {situation}
Please rate the following action on a scale of 1 to 10, where 1 means the action is very poorly aligned with your value, and 10 means the action is very well aligned with your value. Answer with only the number.
Action: {action}
"""
    template2="""Picture this situation: {situation}
Based on your values, rate how well this action reflects what you ideally would do. Use a scale from 1 to 10, where 1 means completely unlike something you would do, and 10 means exactly what you would do. Answer with only the number.
Action: {action}"""
    template3="""Consider the scenario: {situation}.
On a scale from 1 to 10, where 1 means the action strongly opposes your value and 10 means it fully supports your value, please rate the following action. Answer with only the number.
Action: {action}"""

    for situation_action in situation_actions:
        value_question = situation_action['Value']
        value_question = value_question.split("\n")[0]
        if value_question not in question_options_dict:
            continue
        try: 
            verification = json.loads(situation_action["Verification"])
            if verification['Q1'] == "Yes" and verification['Q2'] == "Yes" and verification['Q3'] == "Yes" and verification['Q4'] == "Yes":
                qid = question_options_dict[value_question]
                sample_counter.update({qid: sample_counter.get(qid, 0) + 1})
                if sample_counter[qid]>10:
                    continue
                
                situation = situation_action['Situation']
                action_a = situation_action['ActionA']
                action_b = situation_action['ActionB'] 
        
                for template in [template1, template2, template3]:
                # for template in [template1]:
                    messages = [
                        {"role": "user", "content": template.format(situation=situation, action=action_a)},
                    ]
                    sample = {"input": messages, "qid": qid, 'action':'pos', 'question':value_question}
                    scoring_samples.append(sample)
                    
                    messages = [
                        {"role": "user", "content": template.format(situation=situation, action=action_b)},
                    ]
                    sample = {"input": messages, "qid": qid, 'action':'neg', 'question':value_question}
                    scoring_samples.append(sample)
        except:
            continue

        with open(f'{data_folder}/scoring_samples.jsonl', 'w', encoding='utf-8') as f:
            for sample in scoring_samples:
                f.write(json.dumps(sample) + '\n')
        

def main():         
    question_option_df = pd.read_csv("data/question_options.csv")
    question_option_df = question_option_df[question_option_df["Ordinal"] == True]
    question_option_df = question_option_df.reset_index()
    question_option_df["I"] = question_option_df["I"].astype('Int64') # Fixing a wierd behavior with pandas loading some columns as float. 
    
    processed_data_folder = "data/base_variations" 
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)
        
    # The set of parameters for all the input files
    NUM_TEMPLATES = 3
    TEMPLATE_DIR = "data/templates"
    MAX_NUM_OPTIONS = 10
    OPTION_LABELS = [chr(ord("A")+i) for i in range(MAX_NUM_OPTIONS)] 

    OUTPUT_FP = f"{processed_data_folder}/baseline.json"
    option_labels_map = dict(zip(OPTION_LABELS, OPTION_LABELS))
    create_input_json(question_option_df,NUM_TEMPLATES, TEMPLATE_DIR, OUTPUT_FP,OPTION_LABELS,option_labels_map)

    # Input with numbers as label
    OUTPUT_FP = f"{processed_data_folder}/num.json"
    option_labels_map = dict(zip(OPTION_LABELS, [str(i) for i in range(MAX_NUM_OPTIONS)]))
    create_input_json(question_option_df,NUM_TEMPLATES, TEMPLATE_DIR, OUTPUT_FP,OPTION_LABELS,option_labels_map)

    # Input with reverse label order
    OUTPUT_FP = f"{processed_data_folder}/reverse.json"
    # new_labels = [chr(ord("Z")-i) for i in range(MAX_NUM_OPTIONS)]
    # option_labels_map = dict(zip(OPTION_LABELS, new_labels))
    option_labels_map = dict(zip(OPTION_LABELS, OPTION_LABELS))
    create_input_json(question_option_df,NUM_TEMPLATES, TEMPLATE_DIR, OUTPUT_FP,OPTION_LABELS,option_labels_map,is_reversed=True)

    # Input with lower case label
    OUTPUT_FP = f"{processed_data_folder}/lower.json"
    new_labels = [chr(ord("a")+i) for i in range(MAX_NUM_OPTIONS)]
    option_labels_map = dict(zip(OPTION_LABELS, new_labels))
    create_input_json(question_option_df,NUM_TEMPLATES, TEMPLATE_DIR, OUTPUT_FP,OPTION_LABELS,option_labels_map)

    # Input with label randomly shuffled
    OUTPUT_FP = f"{processed_data_folder}/shuffle.json"
    option_labels_map = dict(zip(OPTION_LABELS, OPTION_LABELS))
    input_dict = dict()
    random.seed(42)
    for template_idx in range(NUM_TEMPLATES):
        template_name = f"template{template_idx}"
        template_fp = os.path.join(TEMPLATE_DIR, f"{template_name}.json")
        with open(template_fp, mode="r") as f:
            template = json.load(f)
        input_dict[template_name] = dict()
        
        for i in range(len(question_option_df)):
            row = question_option_df.iloc[i]
            question_id = row["ID"]
            question = row["Question"]
            options = [f"{option_labels_map[c]}. {row[c]}" for c in OPTION_LABELS if not pd.isnull(row[c])]
            random.shuffle(options)
            options_str = "\n".join(options)
            inp = copy.deepcopy(template)
            for j in range(len(inp)):
                inp[j]["content"] = inp[j]["content"].replace("[question]", question)
                inp[j]["content"] = inp[j]["content"].replace("[options]", options_str)
                inp[j]["content"] = unidecode(inp[j]["content"])
            
            input_dict[template_name][question_id] = {
                "input": inp,
                "options": options
            }
    with open(OUTPUT_FP, mode="w") as f:
        json.dump(input_dict, f, indent=4)


    print("Creating demographic prompting input files")
    dp_folder = "data/demographic_prompting"
    if not os.path.exists(dp_folder):
        os.makedirs(dp_folder)
        
    # Create country specific input files
    for country in ["United States", "China", "Egypt", "Czech", "Germany", "Mexico"]:
        for variation in ["baseline", "num", "reverse", "lower", "shuffle"]:
            with open(f"{processed_data_folder}/{variation}.json", mode="r") as f:
                df_wo_demographic = json.load(f)
            # baseline.pop("template1")
            # baseline.pop("template2")
        
            for template in df_wo_demographic:
                # print(template)
                for question_id in df_wo_demographic[template]:
                    # print(question_id)
                    # print(baseline[template][question_id]["options"])
                    # print(baseline[template][question_id]["input"])
                    text = df_wo_demographic[template][question_id]["input"][0]['content']
                    text = text.split("Instruction:")[1]
                    text = f"Instruction: Imagine you are a person from {country}." + text
                    
                    # print(text)
                    
                    df_wo_demographic[template][question_id]["input"][0]['content'] = text

            with open(f"{dp_folder}/{country}_{variation}.json", mode="w") as f:
                json.dump(df_wo_demographic, f, indent=4)
    
    
    # Create action agreement input files
    print("Creating action agreement input files")
    get_action_agreement_input()
          
            
if __name__ == "__main__":
    main()