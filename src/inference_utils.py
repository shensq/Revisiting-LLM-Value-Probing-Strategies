"""
This module is called by the run_inference.py script to get the inference results for the given questionare.
"""
import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from credentials import access_token
import copy

def get_predictions(model, inputs):
    """
    Encode the sentence using the tokenizer and return the model predictions.
    Params:
        model: the model to get predictions from
        inputs: the tensor of token_ids
    Returns:
        predictions: the outputted token logits
    """

    with torch.no_grad():
        outputs = model(inputs)
        predictions = outputs[0]
    return predictions

def get_option_logits(model, inputs, option_token_ids):
    """
    Get the next token candidates.
    Params:
        predictions: the token logits
        option_token_ids: tensor of token ids of answer choices
    Returns:
        option_logits: the logits of each answer choice
    """
    with torch.no_grad():
        outputs = model(inputs.input_ids)
        predictions = outputs[0]
    next_token_candidates_tensor = predictions[0, -1, :]
    option_logits = next_token_candidates_tensor[option_token_ids]
    return option_logits.tolist()
    



def get_results(input_dict, model, model_name, num_return_sequences, tokenizer, device):
    """
    Gets the results of all 3 inference methods for the given model and country
    Params:
        input_dict: the input dictionary describing the inputs for each question and template
        model: the model to do inference with
        model_name: the model name
        num_return_sequences: the number of strings to create with free text generation 
        tokenizer: the tokenizer for the given model
    Returns:
        input_dict: the updated input dictionary with inference results from all 3 methods
    """
    # iterate through each template
    for template_name in input_dict:
        # iterate through each question
        for question_id in tqdm(input_dict[template_name]):
            input_messages = input_dict[template_name][question_id]["input"]
            options = input_dict[template_name][question_id]["options"]
    
            # Concatenate the system message to the first user message for Mistral
            if "Mistral" in model_name:
                input_messages[1]["content"] = input_messages[0]["content"] + "\n" + input_messages[1]["content"]
                input_messages.pop(0)
            
            # Applies chat template for instruction-tuned model
            if "instruct" in model_name.lower():
                # Continue final message if assistant is the last role
                if input_messages[-1]["role"]=="assistant": 
                    inputs = tokenizer.apply_chat_template(input_messages, tokenize=True, return_dict=True, continue_final_message=True, return_tensors="pt").to(device)
                else:
                    inputs = tokenizer.apply_chat_template(input_messages, tokenize=True, return_dict=True, add_generation_prompt=True, return_tensors="pt").to(device)
            else:
                # concate all messages into one string for non-instruction-tuned models
                content = "\n".join([entry["content"] for entry in input_messages])
                inputs = tokenizer(content, return_tensors="pt").to(device)
            input_length = inputs.input_ids.shape[1]
            
            
            # get the token ids of the answer choices
            option_labels = [x.split(".")[0] for x in options]
            option_token_ids = tokenizer.convert_tokens_to_ids(option_labels)
            # get the answer choice token logits for next-token method
            option_logits = get_option_logits(model, inputs, option_token_ids)
            
            # get the token ids of the answer choices with space for backup 
            option_labels_space = [" " + label for label in option_labels]
            option_labels_space = [tokenizer.tokenize(x)[0] for x in option_labels_space] # assume the token exist
            option_token_ids_space = tokenizer.convert_tokens_to_ids(option_labels_space)
            option_logits_space = get_option_logits(model, inputs, option_token_ids_space)

            # get generated strings. outputs.scores is list of length output_length, each element is a tensor of size num_return_sequences x vocab_size            
            outputs_sampling = model.generate(**inputs, do_sample=True, temperature=1.0, max_new_tokens=64, num_return_sequences=num_return_sequences)
            # print(outputs_sampling)
            generated_text = tokenizer.batch_decode(outputs_sampling[:, input_length:], skip_special_tokens=True)
            
            input_dict[template_name][question_id]["option_logits"] = option_logits
            input_dict[template_name][question_id]["generated_str"] = generated_text
            input_dict[template_name][question_id]["option_logits_space"] = option_logits_space
            
            
            # iterate through each option choice and calculate perplexity score
            full_answer_perplexities = list()
            for option in options:
                inp_tokens = inputs["input_ids"]
                num_inp_tokens = inp_tokens.shape[-1]

                # append answer choice to the model's response
                inp_with_option = copy.deepcopy(input_messages)
                if template_name == "template1":
                    inp_with_option[-1]["content"] += f" {option}"
                else:
                    inp_with_option.append({"role": "assistant", "content": option})

                # applies chat template for instruction-tuned model
                if "instruct" in model_name.lower():
                    # do not add extra generation prompt
                    token_dict_with_option = tokenizer.apply_chat_template(inp_with_option, tokenize=True, return_dict=True, add_generation_prompt=False, return_tensors="pt").to(device) 
                else:
                    content = "\n".join([entry["content"] for entry in inp_with_option])
                    token_dict_with_option = tokenizer(content, return_tensors="pt").to(device) 
                all_tokens = token_dict_with_option["input_ids"]

                # compute perplexity/sequence score
                predictions = get_predictions(model, all_tokens) # 1 x leq_len x vocab_size
                logits = predictions[0, num_inp_tokens-1:-1, :] # output_len x vocab_size. i-th in prediction means the token after seeing i tokens
                true_next_token_ids = all_tokens[0, num_inp_tokens:] # output_len
                ce = torch.nn.functional.cross_entropy(logits, true_next_token_ids)
                perplexity = torch.exp(ce).item()
                full_answer_perplexities.append(perplexity)

            # update the input dict for the given template and question id
            input_dict[template_name][question_id]["full_answer_perplexities"] = full_answer_perplexities
            
            
    return input_dict