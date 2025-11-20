"""Legacy single-process FedAvg baseline.

Prefer running `orchestrator.py` + `client.py` for FCCL-aware simulations.
"""

from __future__ import annotations

import os
import random

import torch

from config import CLIENTS_PER_ROUND, FED_ROUNDS, NUM_CLIENTS, SEED
from training import load_client_data, local_dpo_train, local_sft_train
from utils import (
    average_adapter_states,
    device,
    extract_lora_state_dict,
    get_peft_model_and_tokenizer,
    load_tokenizer_and_base,
    set_lora_state_dict,
)

random.seed(SEED)
torch.manual_seed(SEED)


def client_procedure(client_id: int, global_adapter_state: dict, phase: str = "sft"):
    """Run one local update for the legacy single-process baseline."""
    tokenizer, model = get_peft_model_and_tokenizer()
    if global_adapter_state:
        set_lora_state_dict(model, global_adapter_state)
    sft_data, va_data = load_client_data(client_id)
    if phase == "sft":
        model, avg_loss = local_sft_train(model, tokenizer, sft_data)
    else:
        _, ref_model = load_tokenizer_and_base()
        ref_model.to(device)
        model, avg_loss = local_dpo_train(model, tokenizer, va_data, ref_model)
    adapter_state = extract_lora_state_dict(model)
    return adapter_state, avg_loss


def run_federated():
    """Simplified FedAvg loop for experimentation without Docker/multi-process."""
    _, temp_model = get_peft_model_and_tokenizer()
    global_adapter_state = extract_lora_state_dict(temp_model)

    for r in range(FED_ROUNDS):
        clients = list(range(NUM_CLIENTS))
        selected = random.sample(clients, k=min(CLIENTS_PER_ROUND, len(clients)))
        local_states = []
        losses = []
        for cid in selected:
            adapter_state, loss = client_procedure(cid, global_adapter_state, phase="sft")
            local_states.append(adapter_state)
            losses.append(loss)
        global_adapter_state = average_adapter_states(local_states)
        print(f"[Legacy] FedIT round {r}: avg loss {sum(losses)/len(losses) if losses else 0.0:.4f}")

    _, model = get_peft_model_and_tokenizer()
    set_lora_state_dict(model, global_adapter_state)
    torch.save(model.state_dict(), os.path.join("logs", "ref_model_sft.pth"))

    for r in range(FED_ROUNDS):
        clients = list(range(NUM_CLIENTS))
        selected = random.sample(clients, k=min(CLIENTS_PER_ROUND, len(clients)))
        local_states = []
        losses = []
        for cid in selected:
            adapter_state, loss = client_procedure(cid, global_adapter_state, phase="va")
            local_states.append(adapter_state)
            losses.append(loss)
        global_adapter_state = average_adapter_states(local_states)
        print(f"[Legacy] FedVA round {r}: avg loss {sum(losses)/len(losses) if losses else 0.0:.4f}")

    _, final_model = get_peft_model_and_tokenizer()
    set_lora_state_dict(final_model, global_adapter_state)
    os.makedirs("logs", exist_ok=True)
    torch.save(final_model.state_dict(), "logs/global_final.pth")
    print("Saved global model (with LoRA adapters) to logs/global_final.pth")


if __name__ == "__main__":
    run_federated()

