
"""
my commands:

Kaggle (HH-RLHF run):
    python scripts/train_dpo.py --dataset hh --sft_path /kaggle/input/<sft-output>/outputs/sft/best \
        --output_dir /kaggle/working/outputs/dpo_hh

Kaggle (UltraFeedback run):
    python scripts/train_dpo.py --dataset uf --sft_path /kaggle/input/<sft-output>/outputs/sft/best \
        --output_dir /kaggle/working/outputs/dpo_uf

Smoke test (add to either):
    --train_size 64 --val_size 16 --grad_accum 8 --max_steps 4 --eval_every 2 --log_every 1
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from alignment.data import build_dpo_batch, format_prompt
from alignment.losses import dpo_loss_from_logps, move_batch, response_logprobs_from_batch

DATASETS = {
    # name: (Hugging Face dataset id, train split, validation split)
    "hh": ("Anthropic/hh-rlhf", "train", "test"),
    "uf": ("HuggingFaceH4/ultrafeedback_binarized", "train_prefs", "test_prefs"),
}


##for args im just using the manual; ty for giving the specs :)
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--sft_path", required=True, help="SFT best/ checkpoint: initial policy AND frozen reference")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--train_size", type=int, default=15000)
    p.add_argument("--val_size", type=int, default=1000)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--max_prompt_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--optimizer", choices=["adamw", "rmsprop"], default="adamw")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=50, help="optimizer steps between validations")
    p.add_argument("--log_every", type=int, default=10, help="optimizer steps between train logs")
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

#just turning raw dataset rows into (instruction, chosen, rejected) here
HUMAN_TAG = "\n\nHuman:"
ASSISTANT_TAG = "\n\nAssistant:"

def split_hh_text(text):
    num_human_turns = text.count(HUMAN_TAG)            
    num_assistant_turns = text.count(ASSISTANT_TAG)    
#how many times "\n\nHuman:" shows up and  how many times "\n\nAssistant:" shows up
    is_single_turn = num_human_turns == 1 and num_assistant_turns == 1  ##exactly one ask + one answer
    if not is_single_turn: 
        return None
    ##multi turn convo filter these out
    split_parts = text.split(ASSISTANT_TAG)            
    before_assistant = split_parts[0]                  
    after_assistant = split_parts[1]  ##im cutting at "\n\nAssistant:" -> [human part, assistant part]                 
    human_part_no_tag = before_assistant.replace(HUMAN_TAG, "")  
    instruction = human_part_no_tag.strip()            
    response = after_assistant.strip()                 

    return (instruction, response)  ##same (instruction, response) shape as my SFT data


def hh_row_to_example(row):
    """row has "chosen" and "rejected" strings that share the same Human turn."""
    chosen_text = row["chosen"]                       
    rejected_text = row["rejected"]       
    ##convo strings w/ good and bad assistant replies i extracted
    chosen_parts = split_hh_text(chosen_text)         
    rejected_parts = split_hh_text(rejected_text)      
    ##IMP to know (instruction, rejected/chosen reply) 
    if chosen_parts is None or rejected_parts is None: 
        return None
    ##either side is multi turn -> skip whole pair
    chosen_instruction, chosen_response = chosen_parts        
    rejected_instruction, rejected_response = rejected_parts  

    same_question = chosen_instruction == rejected_instruction  
    if not same_question:                              
        return None
    ##imp note2self:
    ### both sides MUST answer the same prompt and shouldnt happen in HH, but if it does the pair is meaningless
    return (chosen_instruction, chosen_response, rejected_response)  
    ##(instruction, chosen, rejected) is the same shape for hh/ uf


def uf_row_to_example(row):
    chosen_messages = row["chosen"]                   
    rejected_messages = row["rejected"]   
    ##[user msg, assistant BAD reply] and [user msg, assistant GOOD reply] i grabbed
    chosen_is_single_turn = len(chosen_messages) == 2     
    rejected_is_single_turn = len(rejected_messages) == 2 
    if not chosen_is_single_turn or not rejected_is_single_turn:  
        return None
    ##exactly user + assistant and same check on the other side and filtering out longer convos
    chosen_reply_msg = chosen_messages[1]             
    rejected_reply_msg = rejected_messages[1]         
    both_are_assistant = chosen_reply_msg["role"] == "assistant" and rejected_reply_msg["role"] == "assistant"  
    ##now im not sure may be hidden test case but doing to make sure idx 1 really is the answer
    if not both_are_assistant:
        return None

    instruction = row["prompt"]                      
    chosen_response = chosen_reply_msg["content"]      ##just the text of the good reply
    rejected_response = rejected_reply_msg["content"]  ##just the text of the bad reply and good reply. no parsing i think needed here? will do test maybe?
    
    return (instruction, chosen_response, rejected_response) 
##same (instruction, chosen, rejected) shape as hh_row_to_example



##just defining for my ease taht "hh" -> "Anthropic/hh-rlhf", "uf" -> "HuggingFaceH4/ultrafeedback_binarized"

def load_preference_examples(dataset_key, split, size, seed):
    """Return a list of (instruction, chosen, rejected) tuples, `size` long."""
    dataset_id = DATASETS[dataset_key][0]              
    row_to_example = hh_row_to_example if dataset_key == "hh" else uf_row_to_example  
    ##pick the right parser, rest of code doesnt care which
    full_dataset = load_dataset(dataset_id, split=split)  ##just downloading (or load from cache if already downloaded once)
    shuffled_dataset = full_dataset.shuffle(seed=seed)    
    ##fixed seed -> same 15k pairs every run -> reproducible otherwise i dont think would be fair for dpo and sft
    kept_examples = []
    skipped_count = 0
    for row in shuffled_dataset:
        one_example = row_to_example(row)              
        if one_example is None:                        
            skipped_count += 1
            continue
        instruction, chosen_response, rejected_response = one_example  
        has_empty_text = (instruction.strip() == ""
                          or chosen_response.strip() == ""
                          or rejected_response.strip() == "") 
        if has_empty_text:
            skipped_count += 1
            continue
        nothing_to_prefer = chosen_response == rejected_response  
        ##identical answers -> DPO gap is always 0, pure noise so just rejecting simply ez
        if nothing_to_prefer:
            skipped_count += 1
            continue
        kept_examples.append(one_example)              
        if len(kept_examples) >= size:                 
        ##got 15k (or 1k for val) -> stop early hehe
            break

    print(f"[{dataset_key}/{split}] kept {len(kept_examples)} pairs, skipped {skipped_count}")  ##filtering stats for the report ima cook hard
    return kept_examples


##this is just turning (instruction, chosen, rejected) into {"chosen": batch, "rejected": batch}
def collate_dpo_rows(rows, tokenizer, max_length, max_prompt_length):
    prompts = []                                       
    chosen_responses = []
    rejected_responses = []
    #rows = list of (instruction, chosen, rejected), batch_size of them from dataloader grabbed
    for instruction, chosen_response, rejected_response in rows:  
        one_prompt = format_prompt(instruction)        
        prompts.append(one_prompt)
        chosen_responses.append(chosen_response)
        rejected_responses.append(rejected_response)
        ##SAME template as SFT -> "### Instruction:\n...\n\n### Response:\n"

    dpo_batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses,max_length, max_prompt_length)  
    #my data.py is like 2 sft-style batches, prompt cut to 256 from the left
    return dpo_batch                                  


#DPO fwd pass on a ready batch (policy + frozen reference) littry just returns loss, chosen_rewards, rejected_rewards

def dpo_forward(policy, reference, dpo_batch, beta, policy_device, ref_device, use_amp):  
    chosen_batch = dpo_batch["chosen"]               
    rejected_batch = dpo_batch["rejected"]    
    #{input_ids, attention_mask, labels} w/ the GOOD repl andBAD reply
    chosen_on_policy = move_batch(chosen_batch, policy_device)     
    rejected_on_policy = move_batch(rejected_batch, policy_device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):  
        policy_chosen = response_logprobs_from_batch(policy, chosen_on_policy)      
        policy_rejected = response_logprobs_from_batch(policy, rejected_on_policy) 
        ###log pi_theta(y_w|x), (B,), HAS grads and log pi_theta(y_l|x), (B,), HAS grads
    chosen_on_ref = move_batch(chosen_batch, ref_device)            
    rejected_on_ref = move_batch(rejected_batch, ref_device)
    with torch.no_grad():                              
        ref_chosen = response_logprobs_from_batch(reference, chosen_on_ref)         
        ref_rejected = response_logprobs_from_batch(reference, rejected_on_ref)     
##frozen anchor -> no graph, saves memory and log pi_ref(y_w|x) log pi_ref(y_l|x)
    ref_chosen = ref_chosen.to(policy_device)          
    ref_rejected = ref_rejected.to(policy_device)

    loss, chosen_rewards, rejected_rewards = dpo_loss_from_logps(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)  ##my losses.py fn, ORDER matters: pc, pr, rc, rr
    return loss, chosen_rewards, rejected_rewards      ##rewards needed for accuracy + margin in logs/val


#now just easy stuff gathering my validation metrics

@torch.no_grad()
def evaluate(policy, reference, val_loader, beta, policy_device, ref_device, use_amp):
    policy.eval()  ##dropout off so val numbers are deterministic for now
    all_losses = [] # im keeping one loss per val batch and chosen rewards and rejected rewards, one tensor per batch
    all_chosen = []                                    
    all_rejected = []                                
    for dpo_batch in val_loader:
        batch_loss, chosen_rewards, rejected_rewards = dpo_forward(
            policy, reference, dpo_batch, beta, policy_device, ref_device, use_amp)  ##same fwd as training, no grads here
        all_losses.append(batch_loss.item())           
        all_chosen.append(chosen_rewards.float().cpu())     
        ##(B,) -> cpu so gpu memory doesnt pile up on kaggle </3
        all_rejected.append(rejected_rewards.float().cpu())
    chosen = torch.cat(all_chosen)    ##gluing batches ONE reward per val pair, shape (1000,)
    rejected = torch.cat(all_rejected)                 
    ##same, lines up pair by pair w/ chosen

    chosen_wins = chosen > rejected                  
    reward_accuracy = chosen_wins.float().mean().item()     ##fraction of True -> e.g. 0.62
    reward_margin = (chosen - rejected).mean().item()##avg gap r_w - r_l, positive = learning the right way
    mean_loss = sum(all_losses) / max(len(all_losses), 1)   ##avg val DPO loss, starts log 2ish

    policy.train()                                     

    val_metrics = {
        "loss": mean_loss,
        "reward_accuracy": reward_accuracy,
        "reward_margin": reward_margin,
        "chosen_reward": chosen.mean().item(),         
        "rejected_reward": rejected.mean().item(),    
    }
    return val_metrics

# Main (pre-filled except the training step)
def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    if torch.cuda.is_available():
        policy_device = torch.device("cuda:0")
        ref_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    else:        
        policy_device = ref_device = torch.device("cpu")
    use_amp = policy_device.type == "cuda"
    print(f"policy on {policy_device}, reference on {ref_device}")     # gonna do devices: policy on cuda:0, frozen reference on cuda:1 (manual gave this memory tip to me)

    # both start as the SAME SFT checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.sft_path)
    policy = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=torch.float32).to(policy_device)
    policy.config.use_cache = False
    ref_dtype = torch.float16 if ref_device.type == "cuda" else torch.float32
    reference = AutoModelForCausalLM.from_pretrained(args.sft_path, torch_dtype=ref_dtype).to(ref_device)
    reference.eval()
    reference.requires_grad_(False)   # frozen never updated, never builds gradients so im savin some time

    # data gathering
    _, train_split, val_split = DATASETS[args.dataset]
    train_examples = load_preference_examples(args.dataset, train_split, args.train_size, args.seed)
    val_examples = load_preference_examples(args.dataset, val_split, args.val_size, args.seed)
    print(f"train={len(train_examples)} val={len(val_examples)}")
    collate = lambda rows: collate_dpo_rows(rows, tokenizer, args.max_length, args.max_prompt_length)
    train_loader = DataLoader(train_examples, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_examples, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    # optimizer: small FIXED learning rate, no schedule here
    if args.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(policy.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)  # every epoch gets one optimizer step for its last smaller leftover group too
    total_steps = steps_per_epoch * args.epochs                       # this is the actual number of optimizer updates the loop can make
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"optimizer steps={total_steps}")

    best_acc = -1.0
    step = 0
    window = {"loss": 0.0, "chosen": [], "rejected": [], "count": 0}   # train stats since last log
    start = time.time()
    policy.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for micro_step, dpo_batch in enumerate(train_loader, start=1):

            ##usually i collect 32 mini batches before updating the model, but the last group in an epoch can be smaller than 32
            ##for example if there are 100 batches, the groups are 32 + 32 + 32 + 4, so that final group still needs its own update
            current_group_start = ((micro_step - 1) // args.grad_accum) * args.grad_accum
            current_group_size = min(args.grad_accum, len(train_loader) - current_group_start)

            loss, chosen_rewards, rejected_rewards = dpo_forward(policy, reference, dpo_batch, args.beta, policy_device, ref_device, use_amp)  
            ##4 fwd passes, policy ones have grads
            scaler.scale(loss / current_group_size).backward()
            ##divide by the number of batches in THIS group, so a last group of 4 is averaged over 4 instead of pretending it had 32
            window["loss"] += loss.item() ##for the train log every 10 steps
            window["count"] += 1
            window["chosen"].append(chosen_rewards.float().cpu())      ##collect rewards -> train reward accuracy in logs
            window["rejected"].append(rejected_rewards.float().cpu())

            finished_full_group = micro_step % args.grad_accum == 0
            finished_last_group = micro_step == len(train_loader)
            if not finished_full_group and not finished_last_group:
                continue

            # im only using one prefilled optimizer here
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                chosen = torch.cat(window["chosen"])
                rejected = torch.cat(window["rejected"])
                log_metrics(metrics_path, {
                    "step": step, "split": "train",
                    "loss": window["loss"] / window["count"],
                    "reward_accuracy": (chosen > rejected).float().mean().item(),
                    "reward_margin": (chosen - rejected).mean().item(),
                    "grad_norm": float(grad_norm),
                    "elapsed_min": round((time.time() - start) / 60, 2),
                })
                window = {"loss": 0.0, "chosen": [], "rejected": [], "count": 0}

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(policy, reference, val_loader, args.beta, policy_device, ref_device, use_amp)
                improved = val["reward_accuracy"] > best_acc
                log_metrics(metrics_path, {"step": step, "split": "val", **val, "best": improved})
                if improved: ##manual saying to only keep the checkpoint with the best validation reward accuracy
                    best_acc = val["reward_accuracy"]
                    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "best"))

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    save_checkpoint(policy, tokenizer, os.path.join(args.output_dir, "final"))
    print(f"done: {step} steps, best val reward accuracy {best_acc:.4f}, "
          f"{(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
