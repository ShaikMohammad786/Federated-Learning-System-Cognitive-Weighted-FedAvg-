"""Training utilities shared by orchestrator and clients."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Sequence, Tuple

import torch
import torch.optim as optim
from transformers import PreTrainedTokenizerBase

from config import SFT_LR, SFT_LOCAL_EPOCHS, VA_LR, VA_LOCAL_EPOCHS
from utils import (
    compute_response_only_loss,
    read_jsonl,
    seq_logprob_with_grad,
)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)


def load_client_data(client_id: int) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load the SFT and VA slices assigned to a client."""
    sft_path = os.path.join("data", "clients", f"client_{client_id}_sft.jsonl")
    va_path = os.path.join("data", "clients", f"client_{client_id}_va.jsonl")
    sft = read_jsonl(sft_path) if os.path.exists(sft_path) else []
    va = read_jsonl(va_path) if os.path.exists(va_path) else []
    logger.debug("Loaded %d SFT and %d VA samples for client %d", len(sft), len(va), client_id)
    return sft, va


def local_sft_train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sft_data: Sequence[Dict[str, str]],
) -> Tuple[torch.nn.Module, float]:
    """Run instruction-tuning (FedIT) on a client's local SFT slice."""
    model.train()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=SFT_LR)
    losses: List[float] = []
    for _ in range(SFT_LOCAL_EPOCHS):
        for ex in sft_data:
            instr = ex["instruction"]
            resp = ex["response"]
            loss = compute_response_only_loss(model, tokenizer, instr, resp)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss.item()))
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    logger.info("Finished SFT step with avg loss %.4f over %d samples", avg_loss, len(sft_data))
    return model, avg_loss


def local_dpo_train(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    va_data: Sequence[Dict[str, str]],
    ref_model: torch.nn.Module,
) -> Tuple[torch.nn.Module, float]:
    """Run value-alignment (FedVA/DPO) on a client's local VA slice."""
    model.train()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=VA_LR)
    losses: List[float] = []
    beta = 0.1  # small beta
    for _ in range(VA_LOCAL_EPOCHS):
        for ex in va_data:
            instr = ex["instruction"]
            pref = ex["preferred"]
            disp = ex["dispreferred"]
            seq_pref = f"Instruction: {instr}\nResponse: {pref}"
            seq_disp = f"Instruction: {instr}\nResponse: {disp}"
            lp_pref = seq_logprob_with_grad(model, tokenizer, seq_pref)
            lp_disp = seq_logprob_with_grad(model, tokenizer, seq_disp)
            ref_model.eval()
            with torch.no_grad():
                lp_ref_pref = seq_logprob_with_grad(ref_model, tokenizer, seq_pref).detach()
                lp_ref_disp = seq_logprob_with_grad(ref_model, tokenizer, seq_disp).detach()
            diff = (lp_pref - lp_disp) - (lp_ref_pref - lp_ref_disp)
            loss = -torch.log(torch.sigmoid(beta * diff))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    logger.info("Finished VA/DPO step with avg loss %.4f over %d samples", avg_loss, len(va_data))
    return model, avg_loss

