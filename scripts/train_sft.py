"""
Kaggle (full run) command: python scripts/train_sft.py --output_dir /kaggle/working/outputs/sft

Kaggle (10-step smoke test):
    python scripts/train_sft.py --output_dir /kaggle/working/outputs/sft_smoke \
        --train_size 320 --val_size 16 --max_steps 10 --eval_every 5 --log_every 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from alignment.data import build_sft_batch, format_prompt
from alignment.losses import move_batch, sft_loss_from_batch



# Arguments i jus resused frm manual

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dataset_name", default="HuggingFaceH4/ultrachat_200k")
    p.add_argument("--output_dir", default="outputs/sft")
    p.add_argument("--train_size", type=int, default=30000)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=100, help="optimizer steps between validations")
    p.add_argument("--log_every", type=int, default=10, help="optimizer steps between train-loss logs")
    p.add_argument("--max_steps", type=int, default=None, help="stop early (smoke tests)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_metrics(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(record, flush=True)


def save_checkpoint(model, tokenizer, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


##this is the data part of my script
def to_single_turn(messages):
    ##note 2self-> UltraChat row["messages"] is a list of {"role": ..., "content": ...} dicts.
    ##i just need first turn
    ##need at least a user msg + an assistant reply, otherwise nothing to train on
    num_messages = len(messages)
    if num_messages < 2:
        return None
    ##grab the first exchange only
    first_msg = messages[0]
    second_msg = messages[1]
    ##pull out who said what
    first_role = first_msg["role"]
    second_role = second_msg["role"]
    ##has to be user asks -> assistant answers, anything else (e.g. starts w/ system msg) i skip
    if first_role != "user" or second_role != "assistant":
        return None
    ##the actual text
    instruction = first_msg["content"]
    response = second_msg["content"]
    ##skip blank ones, .strip() so "   " also counts as empty
    instruction_is_empty = instruction.strip() == ""
    response_is_empty = response.strip() == ""
    if instruction_is_empty or response_is_empty:
        return None
    return (instruction, response)


def load_sft_examples(dataset_name, split, size, seed):
    ##download (or load from cache) the whole split, e.g. "train_sft" or "test_sft"
    full_dataset = load_dataset(dataset_name, split=split)
    ##shuffle w/ fixed seed so every run picks the SAME subset -> reproducible
    shuffled_dataset = full_dataset.shuffle(seed=seed)
    ##walk through rows, convert each to single-turn, keep the good ones
    kept_examples = []
    skipped_count = 0
    for row in shuffled_dataset:
        one_example = to_single_turn(row["messages"])
        ##None means the row failed a check in to_single_turn -> skip it
        if one_example is None:
            skipped_count += 1
            continue
        kept_examples.append(one_example)
        ##stop as soon as i have enough, no need to read all 200k rows
        if len(kept_examples) >= size:
            break
    print(f"[{split}] kept {len(kept_examples)} examples, skipped {skipped_count}")
    ##im adding this after going thru the manual report it's asking for filtering choices
    return kept_examples


def collate_sft_rows(rows, tokenizer, max_length):
    prompts = []    ##rows = list of (instruction, response) tuples, batch_size of them, from the dataloader
    responses = []
    for instruction, response in rows:
        one_prompt = format_prompt(instruction)      #wrap raw instruction in my template -> "### Instruction:\n...\n\n### Response:\n"
        prompts.append(one_prompt)
        responses.append(response)
    ##my own data.py does tokenizing + masking + padding
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return batch

##getting the VALIDATION loss here (held-out examples the model never trains on)
@torch.no_grad()
def evaluate(model, val_loader, device, use_amp):
    model.eval() ##eval mode -> turns off dropout so the score im gettin is deterministic
    total_loss = 0.0
    num_batches = 0
    for batch in val_loader:
        batch_on_device = move_batch(batch, device) ##batch is built on cpu by the dataloader, model is on gpu
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):         ##using same fp16 autocast as training so val loss is comparable to train loss
            batch_loss = sft_loss_from_batch(model, batch_on_device)
        total_loss += batch_loss.item() ##.item() turns the 0-dim tensor into a plain python float so i can directly do math here
        num_batches += 1

    #model.train()
    model.train()
    ##IMPORTANT: uncommeting for now to make it back to train mode, otherwise the rest of training runs w/o dropout

    mean_loss = total_loss / max(num_batches, 1) ##max(...,1) so an empty val set can't divide by 0
    return mean_loss



# this the main part with my training step

def main():
    args = parse_args()     ##read cmd line flags which are just PDF hyperparams
    set_seed(args.seed)                      ##same seed -> same shuffle + init every run
    os.makedirs(args.output_dir, exist_ok=True)         ##make output folder if missing
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")       ##one json line per log, survives disconnects
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:  ##save exact settings used
        json.dump(vars(args), f, indent=2)   ##-> for the report's config table im doing this

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  ##gpu 0 on kaggle, cpu on laptop
    use_amp = device.type == "cuda"        ##fp16 autocast only makes sense on gpu w/ training

    tokenizer = AutoTokenizer.from_pretrained(args.model_name) ##qwen tokenizer (has eos + pad itself)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32).to(device)  ##fp32 weights, autocast does fp16 math
    model.config.use_cache = False  ##kv cache only helps generation, not training so i dont want time wasted

    train_examples = load_sft_examples(args.dataset_name, "train_sft", args.train_size, args.seed)  
    val_examples = load_sft_examples(args.dataset_name, "test_sft", args.val_size, args.seed)       
    print(f"train={len(train_examples)} val={len(val_examples)}")         ## check counts printed for me to see things are ok
    collate = lambda rows: collate_sft_rows(rows, tokenizer, args.max_length)  ##dataloader only passes rows, lambda fills the rest
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)      
    val_loader = DataLoader(val_examples, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)   
    ## first i did batches of 2 then shuffledbigger batches ok, no grads

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  
    
    ##adamw, lr 2e-5, wd 0.1 in manual so lemme see
    total_steps = math.ceil(len(train_loader) * args.epochs / args.grad_accum)  
    if args.max_steps is not None:                                         
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(args.warmup_ratio * total_steps)                    
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)  ##warmup then cosine decay to 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)                ##scales loss up so fp16 grads dont round to 0
    print(f"optimizer steps={total_steps} warmup={warmup_steps}")

    best_val = float("inf")                                                ##any real val loss beats infinity so i can keep track of best val this way
    step = 0                  ##optimizer steps taken
    running_loss = 0.0     ##sum of micro batch losses since last log
    running_count = 0 ##how many micro batches in that sum
    start = time.time()##for elapsed time in logs
    model.train() ##train mode is dropout on
    optimizer.zero_grad(set_to_none=True)  ##start w/ clean grads

    for epoch in range(args.epochs):                                       
        ## ima only do 1 epoch like mnaual says
        for micro_step, batch in enumerate(train_loader, start=1):         
            batch = move_batch(batch, device)              

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):  ##fp16 forward pass on T4
                loss = sft_loss_from_batch(model, batch)        ##my per-token response-only loss
            scaler.scale(loss / args.grad_accum).backward()                
            running_loss += loss.item()    ##logging the unscaled undivided loss
            running_count += 1

            if micro_step % args.grad_accum != 0:                          
                continue                                                   

            scaler.unscale_(optimizer)    ##undo the scaling so clipping sees real grad sizes
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  ##cap grad norm at 1.0, returns norm BEFORE clip
            scaler.step(optimizer)                                         
            scaler.update()                                                
            scheduler.step()                                               
            optimizer.zero_grad(set_to_none=True)                         
            step += 1    
            ##one real optimizer step is done up to here i think

            if step % args.log_every == 0:   ##every 10 steps i have train stats written
                log_metrics(metrics_path, {
                    "step": step, "split": "train",
                    "loss": running_loss / running_count,                 
                    "lr": scheduler.get_last_lr()[0],                      
                    "grad_norm": float(grad_norm),                         ##spikes here = instability
                    "elapsed_min": round((time.time() - start) / 60, 2),
                })
                running_loss, running_count = 0.0, 0  ##reset for next window to process here

            if step % args.eval_every == 0 or step == total_steps:         
                val_loss = evaluate(model, val_loader, device, use_amp)    
                improved = val_loss < best_val                       ##new best?
                log_metrics(metrics_path, {"step": step, "split": "val", "loss": val_loss, "best": improved})
                if improved:
                    best_val = val_loss
                    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "best"))  ##overwrite best/, DPO starts from this

            if step >= total_steps:   ##smoke test / last step reached so i break out of epoch loop too
                break
        if step >= total_steps:                                           
            break

    save_checkpoint(model, tokenizer, os.path.join(args.output_dir, "final"))  ##PDF wants best AND final
    print(f"done: {step} steps, best val loss {best_val:.4f}, {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()