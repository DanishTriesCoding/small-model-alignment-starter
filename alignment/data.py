"""Prompt formatting and batch construction for SFT and DPO."""

###basically just (prompt, response) is being turned to tensors

from __future__ import annotations

import torch

IGNORE_INDEX = -100

# will be keeping this across my files now in the PA
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction):
    # input:  "  Name a fruit. " and output: "### Instruction:\nName a fruit.\n\n### Response:\n"
    # upper is just my own ref
    stripped_inst = instruction.strip()
    prompt_to_return = PROMPT_TEMPLATE.format(instruction=stripped_inst)
    return prompt_to_return


def get_pad_token_id(tokenizer):
    ##ok main concept is shorter rows get filled with a pad token until as long as longest one appending 
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        return pad_id
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        return eos_id
    return 0


def tokenize_prompt_response(tokenizer, prompt, response, max_length, max_prompt_length=None):
    ##basically handles one example per exmaple gets out the results just need to make sure padding is applied and i truncate on time i think eos ids will be gone
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) ##false cuz the tokenzier itself can edd end of sentence toekns im regulating that myself
    response_ids =  tokenizer.encode(response, add_special_tokens=False)
    ##im also checking the end of sentence if tokenzier has one just in case
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        response_ids = response_ids + [eos_id] ##both are lists
        
    ##adding this in DPO when PDF limit is 256
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]  ##basically keepin end of prompt
    
    full_ids = prompt_ids + response_ids
    ##now it needs to be added to full labels list but prompt ids may not be enough so to add like a padding here too its uh prompt mask tho of defined ignored index doing this
    prompt_mask = [IGNORE_INDEX] * len(prompt_ids) 
    full_labels = prompt_mask + response_ids
    input_ids = full_ids[:max_length]  
    labels = full_labels[:max_length]
    
    ##truncating since context length constraint 
    return input_ids, labels


def pad_examples(examples, pad_token_id):
    ##this is interesting we're doing padding of different lengths and stacking them together
    lengths = [len(ids) for ids, labs in examples] ##collecting len(ids) for each pair
    width = max(lengths)
    batch_size = len(examples)
    
    #now adding 3 tensors here w/ the padding vals 
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    ## every entry is value not empty or random tensor
    
    attention_mask = torch.zeros((batch_size,width), dtype=torch.long)
    labels = torch.full((batch_size,width),IGNORE_INDEX, dtype=torch.long)
    
    ##i think to copy each row in i need enum loop since 0, first_pair), (1, second_pair) stuff
    for row, (ids,labs) in enumerate(examples):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(labs, dtype=torch.long)
        attention_mask[row, :length]  = 1
    ##i unpack each pair and go in this row from columns 0 to length seting all to 1
    
    batch_to_return = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return batch_to_return
    
    
    


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    ## now things get fun im just calling my helpers i think i jsut
    # i just need to grab each prompt response adn tokenize then collect in a list then pad
    sft_examples = []
    for prompt, response in zip(prompts, responses):
        one_example = tokenize_prompt_response(tokenizer, prompt, response, max_length, max_prompt_length)
        sft_examples.append(one_example)

    pad_id = get_pad_token_id(tokenizer)
    batch = pad_examples(sft_examples, pad_id)
    return batch
       


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length, max_prompt_length=None):
    ##cant i just reuse sftbatch 
    chosen = build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length=max_prompt_length)
    rejected = build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length=max_prompt_length)
    return {"chosen": chosen, "rejected": rejected}  