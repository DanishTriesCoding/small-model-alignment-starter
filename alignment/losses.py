"""Response log-probabilities, SFT loss, and DPO loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from alignment.data import IGNORE_INDEX, build_dpo_batch, build_sft_batch


def response_logprobs_from_batch(model, batch):
    #so first i gotta run model on input ids my attn mask size is B,t,v?
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits
    
    shifted_logits = logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]
    mask = shifted_labels.ne(IGNORE_INDEX)     ##causal so shifting and making the mask of positions to score 
    log_probs = F.log_softmax(shifted_logits.float(), dim =-1)
    safe_labels = shifted_labels.masked_fill(~mask,0)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs * mask        #
    row_logps = token_log_probs.sum(dim=-1)         
    return row_logps
    ##remember the result is (B,)
    

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    # 1. build_sft_batch(...)
    # 2. move tensors to the model's device (hint: next(model.parameters()).device)
    # 3. call response_logprobs_from_batch
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    return response_logprobs_from_batch(model, batch)
    


def sft_loss_from_batch(model, batch):
    ###im gonna go with per token altho manual gives both options of per example too
    row_logps = response_logprobs_from_batch(model, batch)
    num_tokens = batch["labels"][:, 1:].ne(IGNORE_INDEX).sum().clamp_min(1)
    ##shifting by 1 so first label not pred and ignoring the ignore INDEX (!= stuff)
    ##then clamping so i dont get divide by 0 problem
    
    loss = -row_logps.sum() / num_tokens
    return loss


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    ##batch wrapper just converts to batch then same calling
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    batch = {key: value.to(device) for key, value in batch.items()}
    return sft_loss_from_batch(model, batch)


def dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    ##im trying to see if the model being trained i.e. policy increassed its pref
    ## for the good answer compared to where ref started right
    ##teach tensor is of shape batch i understnad
    policy_gap = policy_chosen - policy_rejected   
    ref_gap = ref_chosen - ref_rejected 
    logits = beta*(policy_gap - ref_gap)
    loss = -F.logsigmoid(logits).mean()
    chosen_rewards   = beta * (policy_chosen - ref_chosen).detach()
    rejected_rewards = beta * (policy_rejected - ref_rejected).detach()
    return loss,chosen_rewards, rejected_rewards
    

def move_batch(batch, device):
    ##same dict comprehension as in my wrappers, just reusable
    return {key: value.to(device) for key, value in batch.items()}
    


##For this im gonna make a helper to move batches around for my own convenience
def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    
    ##so ill build my dpo batch and then their branches of chosen adn rejecte are normal sft style
    dpo_batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    #i also need to check device wise?
    policy_device = next(policy_model.parameters()).device
    ref_device = next(reference_model.parameters()).device
    policy_chosen = response_logprobs_from_batch(policy_model, move_batch(dpo_batch["chosen"], policy_device))
    policy_rejected = response_logprobs_from_batch(policy_model, move_batch(dpo_batch["rejected"], policy_device))     
    # i need toget the log probs for the ref model like origina model chosen without gradients?
    with torch.no_grad():
        ref_chosen = response_logprobs_from_batch(reference_model, move_batch(dpo_batch["chosen"], ref_device))
        ref_rejected = response_logprobs_from_batch(reference_model, move_batch(dpo_batch["rejected"], ref_device))
    ##ill just get the dpo loss now and return
    ref_chosen = ref_chosen.to(policy_device)
    ref_rejected = ref_rejected.to(policy_device)
    loss, _, _ = dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
    return loss
    
    
    