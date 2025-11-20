"""Utility helpers for model loading, LoRA adapters, and scoring."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from config import LORA_ALPHA, LORA_R, MAX_LENGTH, MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer_and_base() -> Tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]:
    """Load the base causal LM and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, base_model


def get_peft_model_and_tokenizer() -> Tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]:
    """Wrap the base model with LoRA adapters for fine-tuning."""
    tokenizer, base_model = load_tokenizer_and_base()
    # Enable gradient checkpointing to reduce memory
    try:
        base_model.gradient_checkpointing_enable()
    except Exception:
        pass
    base_model.to(device)
    # prepare model for LoRA (we are not quantizing in mini-demo)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["c_attn", "q_attn", "v_attn", "k_attn"],  # target common attention modules; tiny models vary
        inference_mode=False,
        bias="none",
    )
    peft_model = get_peft_model(base_model, config)
    return tokenizer, peft_model


def read_jsonl(path: str) -> List[Dict]:
    """Read a .jsonl file into memory."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for l in f:
            out.append(json.loads(l))
    return out


def save_adapter_state(adapter_state: Dict[str, torch.Tensor], path: str) -> None:
    """Persist an adapter state dict to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in adapter_state.items()}, path)


def load_adapter_state(path: str) -> Dict[str, torch.Tensor]:
    """Load an adapter state dict from disk."""
    return torch.load(path, map_location=device)


def extract_lora_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    # returns only trainable LoRA params
    sd = model.state_dict()
    lora_sd = {k: v.clone().cpu() for k, v in sd.items() if "lora" in k or "adapter" in k}
    return lora_sd


def set_lora_state_dict(model: torch.nn.Module, lora_sd: Dict[str, torch.Tensor]) -> None:
    # load LoRA params into model (leaving base frozen)
    sd = model.state_dict()
    for k, v in lora_sd.items():
        if k in sd:
            sd[k] = v.to(sd[k].device)
    model.load_state_dict(sd, strict=False)


def average_adapter_states_weighted(
    list_states: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """Weighted average of adapter dicts; defaults to uniform FedAvg if weights are missing/invalid."""
    if not list_states:
        return {}
    if weights is None:
        weights = [1.0] * len(list_states)
    if len(weights) != len(list_states):
        raise ValueError("weights must match number of adapter states")
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    total = weight_tensor.sum()
    if total <= 0:
        weight_tensor = torch.ones_like(weight_tensor) / len(list_states)
    else:
        weight_tensor = weight_tensor / total
    avg = {}
    for k in list_states[0].keys():
        stacked = torch.stack([s[k].to(torch.float32) for s in list_states], dim=0)
        # reshape weights for broadcasting
        w = weight_tensor.view([-1] + [1] * (stacked.ndim - 1))
        avg[k] = torch.sum(stacked * w, dim=0)
    return avg


def average_adapter_states(list_states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Backward-compatible uniform averaging wrapper."""
    return average_adapter_states_weighted(list_states)


def seq_logprob(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> float:
    # compute logprob of the given full sequence (text) under model
    # tokenizes text, computes sum log probabilities of each token given previous tokens
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]  # predict next-token
        target = input_ids[:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # pick token log-probs
        token_logps = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)  # shape (1, seq_len-1)
    seq_logp = token_logps.sum().item()
    return seq_logp


def seq_logprob_with_grad(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logps = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum()


def compute_response_only_loss(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    instruction: str,
    response: str,
) -> torch.Tensor:
    prompt = f"Instruction: {instruction}\nResponse:"
    enc_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    enc_full = tokenizer(prompt + f" {response}", return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc_full["input_ids"].to(device)
    attention_mask = enc_full["attention_mask"].to(device)
    labels = input_ids.clone()
    # mask all tokens up to (and including) the space before response
    response_start = enc_prompt["input_ids"].shape[1] - 1
    labels[:, :response_start] = -100
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    return outputs.loss
