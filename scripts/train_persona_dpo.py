"""TRL DPO training for the funny ahh Ramsay persona preference dataset."""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    p = argparse.ArgumentParser()
    ##small instruction model and the preference file already in repo
    p.add_argument("--model_name", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    p.add_argument("--data_path", default="data/persona_lora/ramsay_style_preferences.json")
    p.add_argument("--output_dir", default="/kaggle/working/outputs/persona_dpo")
    ##data = 220 rows; small sanity training check here just
    p.add_argument("--validation_fraction", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=512)
    ##im going w/ these values for  LoRA run tho def may have to change them l8r i think
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="use a small positive value for a smoke run; -1 runs all configured epochs",
    )
    ##rank/alpha for adapter capacity and also doin dropout for preventing i tjhink overfitting or regularization whatever it is 
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated SmolLM2 linear modules that LoRA should adapt",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()



def set_seed(seed):
    random.seed(seed)                  
    np.random.seed(seed)               
    torch.manual_seed(seed)            
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  
    ##this is pretty stragithforward like just do python and numpy random choices and then torch on  cpu then pytorch on both the gpus


def load_preference_data(data_path):
    with open(data_path, "r", encoding="utf-8") as preference_file:
        raw_rows = json.load(preference_file)
    rows_are_a_list = isinstance(raw_rows, list)
    if not rows_are_a_list:
        raise ValueError("The preference file should contain a JSON list of rows.")
    ##so far just opening json and automatically close w/ with and it's big list then sanity check at end for myself if shape 

    required_fields = ("prompt", "chosen", "rejected")
    cleaned_rows = []
    for row_number, current_row in enumerate(raw_rows, start=1):
        current_row_is_a_dictionary = isinstance(current_row, dict)
        if not current_row_is_a_dictionary:
            raise ValueError(f"Row {row_number} should be a JSON object.")
        cleaned_row = {}
###after sanity checking im enuming inside rwa rows and putting cleaned versions in new list i dont wanna do i nplace and also making dct (per checked row)
        for field_name in required_fields:
            field_value = current_row.get(field_name)

            ##all three values should be text, not a number, list, or missing value
            field_value_is_text = isinstance(field_value, str)
            if not field_value_is_text:
                raise ValueError(
                    f"Row {row_number} is missing a text value for '{field_name}'."
                )
            cleaned_field_value = field_value.strip()
            field_value_is_empty = cleaned_field_value == ""
    ##now ig simple checking each feilds like they nums or not and then ir they there then just trim whitespace
            if field_value_is_empty:
                raise ValueError(
                    f"Row {row_number} has an empty '{field_name}' value."
                )
            cleaned_row[field_name] = cleaned_field_value
        cleaned_rows.append(cleaned_row)
    return cleaned_rows
##this im returning list will be formatted by trl by next func



###so just take each json entry and make entry w/ 3 columns but i already got em gonna keep as text; need shape of dpotrainer
def format_for_trl(preference_rows):
    formatted_preference_rows = []
    for row_number, preference_row in enumerate(preference_rows):
        required_columns = ["prompt", "chosen", "rejected"]
        for column_name in required_columns:
            if column_name not in preference_row:
                raise ValueError(
                    f"Preference example {row_number} is missing the '{column_name}' field."
                )
        prompt_text = preference_row["prompt"].strip()
        chosen_text = preference_row["chosen"].strip()
        rejected_text = preference_row["rejected"].strip()
        if prompt_text == "" or chosen_text == "" or rejected_text == "":
            raise ValueError(
                f"Preference example {row_number} has an empty prompt, chosen answer, or rejected answer."
            )
        ###so im using a new list not in place and i check example by example seeing if columns there and strip white space and see if there is text
        ##below is just new rows made and appended then made into trl form 
        formatted_preference_row = {
            "prompt": [{"role": "user", "content": prompt_text}],
            "chosen": [{"role": "assistant", "content": chosen_text}],
            "rejected": [{"role": "assistant", "content": rejected_text}],
        }
        formatted_preference_rows.append(formatted_preference_row)
    trl_dataset = Dataset.from_list(formatted_preference_rows) ##the dataset.py file giving me this .from_list()
    return trl_dataset
###do i return them together like all comparisons? ill see my preparedatasets to confirm


###now just calling built funcs making my life easy since calls load_pref_data func to read from json then just formats then; 
# next is justtraining and validation dataset
def prepare_datasets(args):
    preference_rows = load_preference_data(args.data_path)
    trl_dataset = format_for_trl(preference_rows)
    number_of_examples = len(trl_dataset)
    if number_of_examples < 2:
        raise ValueError("At least two preference examples are needed to make a train/validation split.")
    if args.validation_fraction <= 0 or args.validation_fraction >= 1:
        raise ValueError("validation_fraction must be greater than 0 and less than 1.")
    ##for sanity keeping presplit nums and checking num examples per and remaining
    dataset_split = trl_dataset.train_test_split(
        test_size=args.validation_fraction,
        seed=args.seed,
        shuffle=True,
    )
## im rsplitting with seeds to make reproducbile
    training_dataset = dataset_split["train"]
    validation_dataset = dataset_split["test"] 
    print(
        f"Persona DPO data: "
        f"train={len(training_dataset)} "
        f"validation={len(validation_dataset)}"
    ) ##im pritning for my own peace of mind to see if sizes are sensible or i gotta fix
    return training_dataset, validation_dataset


##this is just grabbing args and passing to LORAconfig to get lora settings sorted out
def build_lora_config(args):
    lora_rank = args.lora_r
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    target_modules = [
        module_name.strip()
        for module_name in args.target_modules.split(",")
        if module_name.strip()
    ]
    lora_configuration = LoraConfig(r=lora_rank,lora_alpha=lora_alpha,lora_dropout=lora_dropout,target_modules=target_modules,bias="none",task_type="CAUSAL_LM",)
    return lora_configuration


def build_trainer(args, model, tokenizer, train_dataset, eval_dataset, peft_config):
    output_directory = args.output_dir
    number_of_epochs = args.num_train_epochs
    training_batch_size = args.per_device_train_batch_size
    evaluation_batch_size = args.per_device_eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    learning_rate = args.learning_rate
    dpo_beta = args.beta
    warmup_fraction = args.warmup_ratio
    maximum_sequence_length = args.max_length
    random_seed = args.seed
    ##grabbing training settings to pass to dpoconfig for trl to understand
    dpo_training_arguments = DPOConfig(output_dir=output_directory,num_train_epochs=number_of_epochs,per_device_train_batch_size=training_batch_size,per_device_eval_batch_size=evaluation_batch_size,
        ##in hindsight, not spacing per line was NOT a wise thing
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=dpo_beta,
        warmup_ratio=warmup_fraction,
        max_length=maximum_sequence_length,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        max_steps=args.max_steps,
        report_to="none",
        seed=random_seed,
    )
    dpo_trainer = DPOTrainer(model=model,args=dpo_training_arguments,train_dataset=train_dataset,eval_dataset=eval_dataset,processing_class=tokenizer,peft_config=peft_config,)
    return dpo_trainer

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    ##im reading args and setting reproducible seed then load my tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    ###using left padding because TRL's DPOTrainer expects prompts to be padded on the left
    # Load the base language model that the LoRA adapter will be trained on.
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    train_dataset, eval_dataset = prepare_datasets(args)
    peft_config = build_lora_config(args)
    trainer = build_trainer(args=args,model=model,tokenizer=tokenizer,train_dataset=train_dataset,eval_dataset=eval_dataset,peft_config=peft_config,)
    training_output = trainer.train()
###now i load lang model to do lora on and then get my datasets and then get configs which i pass to dpo trainer so far ez
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    training_metrics = training_output.metrics
    trainer.log_metrics("train", training_metrics)
    trainer.save_metrics("train", training_metrics)
    evaluation_metrics = trainer.evaluate()
    trainer.log_metrics("eval", evaluation_metrics)
    trainer.save_metrics("eval", evaluation_metrics)
    trainer.save_state()
###i get training metrics and eval ones too then just save state
    print(f"Persona DPO training finished. Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
