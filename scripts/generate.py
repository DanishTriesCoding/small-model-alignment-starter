"""
some commands for me
    # baseline: base Qwen on AlpacaEval
    python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --generator qwen2.5-0.5b-base \
        --eval_set alpaca --output_path /kaggle/working/outputs/generations/alpaca_base.json

    # SFT on both eval sets
    python scripts/generate.py --model_path <sft>/best --generator sft --eval_set alpaca \
        --output_path /kaggle/working/outputs/generations/alpaca_sft.json
    python scripts/generate.py --model_path <sft>/best --generator sft --eval_set hh \
        --output_path /kaggle/working/outputs/generations/hh_sft.json

    # DPO models on their matching eval set
    python scripts/generate.py --model_path <dpo_uf>/best --generator dpo_uf --eval_set alpaca \
        --output_path /kaggle/working/outputs/generations/alpaca_dpo_uf.json
    python scripts/generate.py --model_path <dpo_hh>/best --generator dpo_hh --eval_set hh \
        --output_path /kaggle/working/outputs/generations/hh_dpo_hh.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import format_prompt

ALPACA_SUBSET_PATH = "eval_sets/alpaca_eval_300_seed42.json"
HH_EVAL_PATH = "eval_sets/hh_rlhf_eval_300_seed42.json"


#args i just used the manual ones wanna play it safe u know
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="HF model id or local checkpoint folder")
    p.add_argument(
        "--adapter_path",
        default=None,
        help="optional PEFT/LoRA adapter folder to attach to model_path",
    )
    p.add_argument("--generator", required=True, help="name stored in every record, e.g. sft / dpo_uf")
    p.add_argument("--eval_set", choices=["alpaca", "hh"], required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--repetition_penalty", type=float, default=1.1)
    p.add_argument("--no_repeat_ngram_size", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only the first N prompts (sanity check)")
    p.add_argument("--alpaca_size", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def save_json(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)   # atomic swap: never leaves a half-written file


#it's just the fixed prompt sets where i download alpacaeval and pick them with fixed seed so i can do for other runs same samples hehe
def build_alpaca_subset(size, seed):
    from huggingface_hub import hf_hub_download
    file_path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")  
    ##downloads once, cached after for my ease and next i just grab the list of the 800ish dicts
    with open(file_path, encoding="utf-8") as f:
        all_rows = json.load(f)                       
    rng = random.Random(seed)                          
    ##my OWN seeded generator, nothing else can mess w/ its state
    picked_rows = rng.sample(all_rows, size)          

    subset = []
    for i, row in enumerate(picked_rows):##i = 0..299 used for id
        one_prompt = {
            "id": f"alpaca-eval-{seed}-{i:04d}",       
            "instruction": row["instruction"],         
            ##the actual question, all i need for generation
            "dataset": "alpaca_eval",     ##tag so i know where it came from
        }
        subset.append(one_prompt)

    return subset  ##list of 300 {"id", "instruction", "dataset"}


def load_eval_prompts(eval_set, alpaca_size, seed):
    ##just gives me dicts for this eval set
    if eval_set == "hh":                               
        with open(HH_EVAL_PATH, encoding="utf-8") as f:
            hh_prompts = json.load(f)                 
        return hh_prompts
    ##HH safety set is already provided in the repo so thats nice

    subset_already_saved = os.path.exists(ALPACA_SUBSET_PATH)  
    ##did an earlier run already pick the 300?
    if subset_already_saved:
        with open(ALPACA_SUBSET_PATH, encoding="utf-8") as f:
            alpaca_prompts = json.load(f)  ##reuse the EXACT same 300 -> fair across base/sft/dpo
        return alpaca_prompts

    alpaca_prompts = build_alpaca_subset(alpaca_size, seed)  ##very first run only: download + pick 300 w/ seed 42
    save_json(ALPACA_SUBSET_PATH, alpaca_prompts)      ##save once so every later run reads this file instead
    print(f"built new AlpacaEval subset -> {ALPACA_SUBSET_PATH}")  
    ##so i notice in the log when this happens
    return alpaca_prompts


###now im just gonna make it into a nice format clean answer
STOP_MARKERS = ["### Instruction:", "### Response:"]   
##template pieces that mean "model started a NEW fake turn"


def clean_output(text):
    cut_position = len(text)                           
    ##keeping everything when o marker found)
    for marker in STOP_MARKERS:
        marker_position = text.find(marker)    ##index where marker starts. I do -1 if not in text
        marker_found = marker_position != -1
        if marker_found and marker_position < cut_position:  ##only moving  cut EARLIER
            cut_position = marker_position     
            ##earliest marker wins according to my logic but need to confirm l8r

    kept_text = text[:cut_position]                    
    cleaned = kept_text.strip()  ## at end i drop leading/trailing spaces + newlines
    return cleaned
    
    
##returning for one pair
@torch.no_grad()   ##inference only needs no grad graph, way less memory for me
def generate_batch(model, tokenizer, instructions, args, device):
    prompts = []
    for instruction in instructions:
        one_prompt = format_prompt(instruction)        
        ##SAME template as SFT/DPO training which is like ### Instruction:\n...\n\n### Response:\n
        prompts.append(one_prompt)

    encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)  
    ##LEFT padded (set in main) so all prompts end at same column
    input_ids = encoded["input_ids"].to(device)        
    ##(B, prompt_len) on gpu
    attention_mask = encoded["attention_mask"].to(device)  
    ##0 on the left pads so model ignores them

    output_ids = model.generate(input_ids=input_ids,attention_mask=attention_mask,max_new_tokens=args.max_new_tokens,            
        do_sample=False,                               
        repetition_penalty=args.repetition_penalty,   
        no_repeat_ngram_size=args.no_repeat_ngram_size,  
        pad_token_id=tokenizer.pad_token_id,           
        eos_token_id=tokenizer.eos_token_id,           
    )                                                 
    ##i cap answer length at 256 tokens, and then get greedy to deterministic
    ##im also following manual here to discourage repeating tokens and not reapting same token pharses and filling finished rows too
    ## and eos and pad token ids 

    prompt_length = input_ids.shape[1]   ##same for every row thanks to left padding
    new_tokens = output_ids[:, prompt_length:]        
    ##only the generated answer im taking
    raw_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)  

    cleaned_outputs = []
    for raw_text in raw_texts:
        cleaned_outputs.append(clean_output(raw_text)) ##cutting fake "### Instruction:" turn + strip
    return cleaned_outputs  ##one answer per instruction, same order
# main pass now with gen loop

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_eval_prompts(args.eval_set, args.alpaca_size, args.seed)
    if args.limit is not None:
        prompts = prompts[: args.limit]
    dataset_name = "alpaca_eval" if args.eval_set == "alpaca" else "hh_rlhf_style_eval"
    print(f"{len(prompts)} prompts from {args.eval_set}")
    #keep anything already generated by an earlier (interrupted) run just in case anything crashes dont wanna rerun from start
    records = []
    if os.path.exists(args.output_path):
        with open(args.output_path, encoding="utf-8") as f:
            records = json.load(f)
    done_ids = {record["id"] for record in records}
    todo = [row for row in prompts if row["id"] not in done_ids]
    print(f"already done: {len(done_ids)}  to generate: {len(todo)}")
    #due to kaggle and stuff to make it not slow im doing fp16 for inference on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"  
    ##left padding use here
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # First load the regular base model. For ordinary SFT/DPO checkpoints,
    # this is also the complete model that we will use for generation.
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype).to(device)

    # A LoRA training output folder contains adapter weights, not a complete
    # standalone language model, so load those weights onto the matching base.
    if args.adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=False,
        )
        model = model.to(device)
        print(f"loaded LoRA adapter from: {args.adapter_path}")

    model.eval()

    start = time.time()
    for batch_start in range(0, len(todo), args.batch_size):
        batch_rows = todo[batch_start: batch_start + args.batch_size]

        instructions = [row["instruction"] for row in batch_rows]  ##just the question text for this batch
        outputs = generate_batch(model, tokenizer, instructions, args, device)
        ##one cleaned answer per question so far

        for row, output in zip(batch_rows, outputs): ##pair each prompt row w/ its answer, same order
            record = {
                "id": row["id"],                       
                "instruction": row["instruction"],
                "output": output,
                "generator": args.generator, ##e.g. "sft", "dpo_uf" -> hidden from judge later
                "dataset": dataset_name,               
            }
            if "category" in row:                      
                ##HH set has categories (harmful_instructions etc.) so keeping for analysis ig in my report
                record["category"] = row["category"]
            if "expected_behavior" in row:
                record["expected_behavior"] = row["expected_behavior"]
            records.append(record)
        save_json(args.output_path, records)   
        #im saving to my json after EVERY batch
        print(f"{len(records)}/{len(prompts)} done, {(time.time() - start) / 60:.1f} min", flush=True)

    ##ig ill shwo some outputs manual keeps saying for a sanity check so i gotta try
    for record in records[:3]:
        print("=" * 80)
        print("INSTRUCTION:", record["instruction"][:300])
        print("-" * 80)
        print("OUTPUT:", record["output"][:600])
    print(f"saved {len(records)} records -> {args.output_path}")


if __name__ == "__main__":
    main()
