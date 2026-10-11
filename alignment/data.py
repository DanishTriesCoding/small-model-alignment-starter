"""Prompt formatting and batch construction for SFT and DPO.

Every sequence is built the same way:

    [prompt tokens] [response tokens] [EOS]   -> truncated to max_length -> right-padded

Labels copy input_ids on response tokens (including EOS) and are -100 on prompt
and padding tokens. Labels are *not* shifted here; the next-token shift happens
when log-probabilities are computed (see alignment/losses.py).
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100

# One plain-text template shared by SFT training, DPO training, and generation.
# The base model (Qwen/Qwen2.5-0.5B) has no instruction format of its own, so a
# simple, explicit template is easier to keep consistent than a chat template.
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def format_prompt(instruction: str) -> str:
    """Wrap a raw instruction in the shared prompt template."""

    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def _pad_token_id(tokenizer) -> int:
    """Pad with the tokenizer's pad token, falling back to EOS, then 0."""

    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    return 0


def tokenize_prompt_response(tokenizer, prompt, response, max_length, max_prompt_length=None):
    """Tokenize one example into unpadded (input_ids, labels) lists.

    The prompt and response are tokenized separately so the prompt/response
    boundary is exact. EOS is appended to the response *before* truncation, so a
    response that hits max_length loses its EOS (as the assignment specifies).

    If max_prompt_length is set, long prompts are cut from the left (keeping the
    end of the prompt, which sits next to the response).
    """

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]

    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def pad_examples(examples, pad_token_id):
    """Right-pad a list of (input_ids, labels) pairs into batch tensors.

    Rows keep their input order. Width is the longest row in the batch.
    """

    width = max(len(input_ids) for input_ids, _ in examples)
    batch_size = len(examples)

    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    labels = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)

    for row, (row_ids, row_labels) in enumerate(examples):
        length = len(row_ids)
        input_ids[row, :length] = torch.tensor(row_ids, dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(row_labels, dtype=torch.long)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_sft_batch(tokenizer, prompts, responses, max_length, max_prompt_length=None):
    """Return {input_ids, attention_mask, labels} with response-only labels."""

    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")

    examples = [
        tokenize_prompt_response(tokenizer, prompt, response, max_length, max_prompt_length)
        for prompt, response in zip(prompts, responses)
    ]
    return pad_examples(examples, _pad_token_id(tokenizer))


def build_dpo_batch(
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    max_length,
    max_prompt_length=None,
):
    """Return {"chosen": batch, "rejected": batch}, each built like an SFT batch.

    Both sides use the identical prompt formatting and truncation, so the only
    difference between row i of "chosen" and row i of "rejected" is the response.
    """

    if not len(prompts) == len(chosen_responses) == len(rejected_responses):
        raise ValueError("prompts, chosen_responses and rejected_responses must have the same length")

    return {
        "chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length, max_prompt_length),
        "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length, max_prompt_length),
    }
