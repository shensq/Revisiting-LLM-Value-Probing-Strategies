"""
This module parse the generated text to extract the answer. 
"""
import os
import json
import re

def extract_answer_pattern(text):
    """Extract the answer label from the text if exists"""
    text = text.strip()
    # Find the match of {option}[.,:\n]
    # pattern = r"[A-Z0-9][,.:\n]"
    pattern = r"[A-Z0-9][,.:\n]|\([A-Z0-9]\)"
    matches = re.findall(pattern, text)

    short_pattern = r"[A-Z]"
    if len(text)<5:
        matches = re.findall(short_pattern, text)

    matches = set(matches)
    if len(matches)==1:
        result = list(matches)[0][0] # take the valid option
    # elif len(matches)==0:
    #     result = None
    else:
        # result = "Multiple" # have multiple conflict answers
        result = text

    return result

def parse_geneartion_pattern(input_file=None):
    if not input_file:
        folder = "./results"
        results_files = []
        for root, dirs, files in os.walk(folder):
            for file in files:
                if "action_score.jsonl" not in file: # skip the action score files
                    results_files.append(os.path.join(root, file))
    else:
        results_files = [input_file]
        
    for file in results_files:
        with open(file, 'r') as f:
            data = json.load(f)
        print(file)
        for template_name in data:
            # print(template_name)
            try:
                for question_id in data[template_name]:
                    sample = data[template_name][question_id]
                    # valid_options = [o.split(".")[0] for o in sample['options']]
                    results = [extract_answer_pattern(s) for s in sample['generated_str']]
                    # print(results)
                    data[template_name][question_id]["parsed_generation"] = results
            except Exception as e:
                print(e,f"Error with {file}")
        with open(file, 'w') as f:
            f.write(json.dumps(data, indent=4))

def parse_generation_classification():
    # TODO: done by external script
    raise NotImplementedError

def main():
    parse_geneartion_pattern()
            
if __name__ == "__main__":
    main()