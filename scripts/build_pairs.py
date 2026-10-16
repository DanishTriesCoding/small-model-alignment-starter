##so after reading the manual i understand this is to build anonymized A/B answer pairs for the model judge

from __future__ import annotations

import argparse
import json
import os
import random



##simple func to read the two generation files and paths and teh seed
def parse_args():
    parser = argparse.ArgumentParser()  
    #this reads cmds after python build_pairs.py called
    first_generation_file_argument = parser.add_argument("--generation_a",required=True,help="JSON file with the first model's answers",)  
    second_generation_file_argument = parser.add_argument("--generation_b",required=True,help="JSON file with the second model's answers",)  
    blind_pairs_output_argument = parser.add_argument("--pairs_output",required=True,help="Where to save the anonymized A/B pairs as JSONL",)  
## i get both model's saved answers and then the blind pairs JSONL is what the judge will read
    model_identity_key_output_argument = parser.add_argument("--key_output",required=True,help="Where to save the separate A/B-to-model mapping as JSONL",)  
    random_seed_argument = parser.add_argument("--seed",type=int,default=42,help="Seed used to randomize which model appears as A or B",)  
    ##im still taking model identity separately for l8r and then same seed for reproducibility
    parsed_arguments = parser.parse_args() 
    return parsed_arguments  
    

###this is just loading a single model's saved records from jsons
def load_generation_file(file_path):
    with open(file_path, "r", encoding="utf-8") as generation_file:
        generation_records = json.load(generation_file) 
    generation_records_are_a_list = isinstance(generation_records, list)
    if not generation_records_are_a_list:
        raise ValueError(f"Generation file '{file_path}' should contain a JSON list of records.") 
    ## i only read no editing and then the generator saves a list so something else means this not expected format
    required_fields = ("id", "instruction", "output", "generator", "dataset")
    for record_number, generation_record in enumerate(generation_records, start=1):
        generation_record_is_a_dictionary = isinstance(generation_record, dict)
        if not generation_record_is_a_dictionary:
            raise ValueError(f"Record {record_number} in '{file_path}' should be a JSON object.")  
    ## each saved answer should be one little dictionary w/ prompt + answer
        for field_name in required_fields:
            field_value = generation_record.get(field_name)
            field_value_is_text = isinstance(field_value, str)
            field_value_is_blank = field_value_is_text and field_value.strip() == ""
            blank_is_allowed = field_name == "output"
            ##an empty answer is still a real model result, judge can just count it as the worse one

            if not field_value_is_text or (field_value_is_blank and not blank_is_allowed):
                raise ValueError(
                    f"Record {record_number} in '{file_path}' needs a non-empty "
                    f"text value for '{field_name}'."
                )
    ##trying to catch missing IDs answers here before they cause confusing problems later
    return generation_records  
    ##returning checked list back


##simply mapping
def index_records_by_id(generation_records, generator_name):
    records_by_prompt_id = {}  ##initilaizing this here to see answer using prompt ID
    for record_number, generation_record in enumerate(generation_records, start=1):
        prompt_id = generation_record["id"]
        recorded_generator_name = generation_record["generator"]
        generator_name_matches = recorded_generator_name == generator_name
        ##simply id of question across diff model files and model label by generatepy
        if not generator_name_matches:
            raise ValueError(f"Record {record_number} says its generator is "f"'{recorded_generator_name}', but I expected '{generator_name}'.")  
        ##catching any wrong model file catched just in case
        prompt_id_is_already_here = prompt_id in records_by_prompt_id
        if prompt_id_is_already_here:
            raise ValueError(f"Prompt ID '{prompt_id}' appears more than once for '{generator_name}'.")  
        ##im only doing one answer per prompt per model to prevent ambiguity then retaining full record
        records_by_prompt_id[prompt_id] = generation_record  
    return records_by_prompt_id 

###thsi matches shared prompt ids then random assignment of answers
def build_blind_pairs(generations_by_model, seed):
    model_names = list(generations_by_model.keys())#2 model labels for this comparison
    number_of_models = len(model_names)
    if number_of_models != 2:
        raise ValueError(f"I need exactly two models to build A/B pairs, but got {number_of_models}.") #1 model against one other
    first_model_name = model_names[0]
    second_model_name = model_names[1]
    first_model_records_by_id = generations_by_model[first_model_name]
    second_model_records_by_id = generations_by_model[second_model_name]
    first_model_prompt_ids = set(first_model_records_by_id.keys())
    second_model_prompt_ids = set(second_model_records_by_id.keys())
    first_only_prompt_ids = sorted(first_model_prompt_ids - second_model_prompt_ids)
    second_only_prompt_ids = sorted(second_model_prompt_ids - first_model_prompt_ids)
    ##both models must have all the same prompts so the comparison is fair, not silently smaller
    if first_only_prompt_ids or second_only_prompt_ids:
        raise ValueError(
            f"The generation files do not contain the same prompt IDs: "
            f"'{first_model_name}' has {len(first_only_prompt_ids)} extra and "
            f"'{second_model_name}' has {len(second_only_prompt_ids)} extra. "
            f"Examples: {first_only_prompt_ids[:3]} / {second_only_prompt_ids[:3]}"
        )

    shared_prompt_ids = sorted(first_model_prompt_ids)
    ##same complete prompt set in both files; sorting gives us repeatable output order
    if len(shared_prompt_ids) == 0:
        raise ValueError("The two generation files do not share any prompt IDs.")
    random_number_generator = random.Random(seed)  # keeping this random order reproducible w/ same seed
    blind_pairs = []
    identity_key = []
    for pair_number, prompt_id in enumerate(shared_prompt_ids, start=1):
        first_model_record = first_model_records_by_id[prompt_id]
        second_model_record = second_model_records_by_id[prompt_id]
        first_model_instruction = first_model_record["instruction"]
        second_model_instruction = second_model_record["instruction"]
        instructions_match = first_model_instruction.strip() == second_model_instruction.strip()
        if not instructions_match:
            raise ValueError(f"Prompt ID '{prompt_id}' has different instructions in the two files.")

        first_dataset_name = first_model_record["dataset"]
        second_dataset_name = second_model_record["dataset"]
        datasets_match = first_dataset_name == second_dataset_name
        if not datasets_match:
            raise ValueError(f"Prompt ID '{prompt_id}' has different dataset labels in the two files.")

        first_category = first_model_record.get("category")
        second_category = second_model_record.get("category")
        categories_match = first_category == second_category
        if not categories_match:
            raise ValueError(f"Prompt ID '{prompt_id}' has different category labels in the two files.")

        pair_id = f"pair-{pair_number:06d}" #using neutral id w/o model name info
        first_model_goes_as_a = random_number_generator.choice([True, False])
        if first_model_goes_as_a:
            response_a = first_model_record["output"]
            response_b = second_model_record["output"]
            model_a_name = first_model_name
            model_b_name = second_model_name
        else:
            response_a = second_model_record["output"]
            response_b = first_model_record["output"]
            model_a_name = second_model_name
            model_b_name = first_model_name
        blind_pair = {
            "pair_id": pair_id,
            "prompt_id": prompt_id,
            "dataset": first_dataset_name,
            "instruction": first_model_instruction,
            "response_a": response_a,
            "response_b": response_b,
        }
        if first_category is not None:
            blind_pair["category"] = first_category  # lets the judge apply safety-aware criteria without exposing model identity
        blind_pairs.append(blind_pair)
        model_identity_record = {
            "pair_id": pair_id,
            "prompt_id": prompt_id,
            "dataset": first_dataset_name,
            "model_a": model_a_name,
            "model_b": model_b_name,
        }
        identity_key.append(model_identity_record)
    return blind_pairs, identity_key 
###2 sepearate json files here



##LITRLLY just saving json
def save_jsonl(file_path, records):
    output_folder = os.path.dirname(file_path)  
    if output_folder == "":
        output_folder = "." 
    #just need to grab filename
    os.makedirs(output_folder, exist_ok=True) ##making folder incase not done before

    temporary_file_path = file_path + ".tmp"##temp before atual filee

    with open(temporary_file_path, "w", encoding="utf-8") as output_file:
        for one_record in records:
            json.dump(one_record, output_file, ensure_ascii=False)
            output_file.write("\n")  # each line is one independent record for the Kaggle judge to resume from

    os.replace(temporary_file_path, file_path) ##i then replace in actual file once processed


def main():
    args = parse_args()  
    #rn im only getting args like the two input+output paths and seed

    generation_a_path = os.path.normcase(os.path.abspath(args.generation_a))
    generation_b_path = os.path.normcase(os.path.abspath(args.generation_b))
    pairs_output_path = os.path.normcase(os.path.abspath(args.pairs_output))
    key_output_path = os.path.normcase(os.path.abspath(args.key_output))
    output_paths_are_different = pairs_output_path != key_output_path
    outputs_do_not_overwrite_inputs = (
        pairs_output_path not in {generation_a_path, generation_b_path}
        and key_output_path not in {generation_a_path, generation_b_path}
    )
    if not output_paths_are_different:
        raise ValueError("--pairs_output and --key_output must be different files.")
    if not outputs_do_not_overwrite_inputs:
        raise ValueError("Output paths must not overwrite either generation input file.")
    #these checks stop me from overwriting my original answers or one output with the other

    first_generation_records = load_generation_file(args.generation_a)
    second_generation_records = load_generation_file(args.generation_b)
    first_file_has_records = len(first_generation_records) > 0
    second_file_has_records = len(second_generation_records) > 0
    if not first_file_has_records or not second_file_has_records:
        raise ValueError("Both generation files need at least one answer record.")
    #sanity check for no model name / make file is empty
    first_generator_name = first_generation_records[0]["generator"]
    second_generator_name = second_generation_records[0]["generator"]
    model_names_are_different = first_generator_name != second_generator_name
    if not model_names_are_different:
        raise ValueError(
            f"Both files say the generator is '{first_generator_name}'. "
            "I need two different model names for an A/B comparison."
        )
    first_records_by_id = index_records_by_id(first_generation_records,first_generator_name,)
    second_records_by_id = index_records_by_id(second_generation_records,second_generator_name,)
    generations_by_model = {first_generator_name: first_records_by_id,second_generator_name: second_records_by_id,}  
    blind_pairs, identity_key = build_blind_pairs(generations_by_model,args.seed,)
    save_jsonl(args.pairs_output, blind_pairs)
    save_jsonl(args.key_output, identity_key)
    print(f"Built {len(blind_pairs)} blind pairs.")
    #### just making sure records match w/ prompt ids
    print(f"Compared: {first_generator_name} vs {second_generator_name}")
    print(f"Blind pairs saved to: {args.pairs_output}")
    print(f"Model identity key saved to: {args.key_output}")
if __name__ == "__main__":
    main()
