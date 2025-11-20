"""Server/orchestrator coordinating FedIT + FedVA with FCCL weighting."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from typing import Dict, List, Tuple

import torch

from config import (
    CLIENTS_PER_ROUND,
    CONTRIBUTION_MODE,
    CONTRIBUTION_SOURCE,
    CONTRIBUTION_TEMPERATURE,
    FED_ROUNDS,
    NUM_CLIENTS,
)
from utils import (
    average_adapter_states_weighted,
    device,
    extract_lora_state_dict,
    get_peft_model_and_tokenizer,
    set_lora_state_dict,
)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)

# Exchange directory shared with clients through Docker volumes.
EXCHANGE = os.environ.get("EXCHANGE_DIR", "/workspace/exchange")
CONTRIB_EPS = 1e-6

# Phase labels and file patterns used throughout the exchange protocol.
PHASE_SFT = "sft"
PHASE_VA = "va"
PHASE_DONE = "done"
PHASE_FILE_FMT = "phase_{round_id}.txt"
GLOBAL_FILE_FMT = "global_round_{round_id}.pt"
CLIENT_FILE_FMT = "client_{client_id}_round_{round_id}.pt"
DONE_FLAG_FMT = "done_{client_id}_{round_id}.flag"
CONTRIB_FILE_FMT = "contrib_client_{client_id}_round_{round_id}.json"


def write_phase(round_id: int, phase: str) -> None:
    """Persist the current phase marker for clients."""
    with open(os.path.join(EXCHANGE, PHASE_FILE_FMT.format(round_id=round_id)), "w", encoding="utf-8") as f:
        f.write(phase)


def write_global(round_id: int, adapter_state: Dict[str, torch.Tensor]) -> None:
    """Broadcast the latest global adapter to clients via disk."""
    torch.save(adapter_state, os.path.join(EXCHANGE, GLOBAL_FILE_FMT.format(round_id=round_id)))


def wait_for_clients(selected: List[int], round_id: int, timeout: int = 86400) -> None:
    """Block until all selected clients upload their done flags."""
    start = time.time()
    while True:
        done_flags = [os.path.exists(os.path.join(EXCHANGE, DONE_FLAG_FMT.format(client_id=cid, round_id=round_id))) for cid in selected]
        if all(done_flags):
            return
        time.sleep(1)
        if time.time() - start > timeout:
            raise TimeoutError("Client wait timeout")


def collect_client_updates(selected: List[int], round_id: int) -> List[Dict[str, torch.Tensor]]:
    """Read adapter updates produced by selected clients."""
    updates = []
    for cid in selected:
        path = os.path.join(EXCHANGE, CLIENT_FILE_FMT.format(client_id=cid, round_id=round_id))
        updates.append(torch.load(path, map_location=device))
    return updates


def read_contributions(selected: List[int], round_id: int) -> Dict[int, Dict]:
    """Load client contribution metadata (`contrib_client_*`) used for weighting."""
    contrib: Dict[int, Dict] = {}
    for cid in selected:
        path = os.path.join(EXCHANGE, CONTRIB_FILE_FMT.format(client_id=cid, round_id=round_id))
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    contrib[cid] = json.load(f)
            except Exception as exc:
                logger.warning("Failed to read contribution file for client %s: %s", cid, exc)
    return contrib


def compute_contribution_weights(selected: List[int], contrib_info: Dict[int, Dict]) -> List[float]:
    """Compute FCCL weights: lower loss / more samples => higher contribution."""
    if not selected:
        return []
    if CONTRIBUTION_MODE != "cognitive":
        # Non-FCCL runs default to uniform FedAvg.
        return [1.0] * len(selected)

    scores = []
    for cid in selected:
        info = contrib_info.get(cid, {})
        loss_val = info.get("avg_loss")
        loss_val = float(loss_val) if loss_val is not None else 0.0
        num_samples = float(info.get("num_samples") or 0.0)
        if CONTRIBUTION_SOURCE == "loss_inverse":
            base = 1.0 / max(loss_val, CONTRIB_EPS)
        elif CONTRIBUTION_SOURCE == "data_size":
            base = num_samples
        elif CONTRIBUTION_SOURCE == "data_loss_combined":
            base = num_samples / max(loss_val, CONTRIB_EPS)
        else:
            base = 1.0
        scores.append(max(base, CONTRIB_EPS))

    temperature = float(CONTRIBUTION_TEMPERATURE)
    if temperature > 0:
        # Temperature controls sharpness of the softmax distribution over scores.
        score_tensor = torch.tensor(scores, dtype=torch.float32) / max(temperature, CONTRIB_EPS)
        weights = torch.softmax(score_tensor, dim=0).tolist()
    else:
        total = sum(scores)
        weights = [s / total if total > 0 else 1.0 / len(scores) for s in scores]
    return weights


def initial_adapter() -> Dict[str, torch.Tensor]:
    """Extract the initial (zero) adapter state from a fresh PEFT model."""
    _, temp_model = get_peft_model_and_tokenizer()
    return extract_lora_state_dict(temp_model)


def _log_progress(payload: Dict) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(os.path.join("logs", "progress.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_phase(
    start_round: int,
    rounds: int,
    phase: str,
    global_adapter: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Execute a FedIT or FedVA phase for a given number of rounds."""
    round_id = start_round
    for _ in range(rounds):
        clients = list(range(NUM_CLIENTS))
        selected = random.sample(clients, k=min(CLIENTS_PER_ROUND, len(clients)))
        write_phase(round_id, phase)
        _log_progress({"round": round_id, "phase": phase, "selected_clients": selected, "event": "started"})
        logger.info("Round %s phase %s started with clients %s", round_id, phase, selected)

        write_global(round_id, global_adapter)
        _log_progress({"round": round_id, "phase": phase, "selected_clients": selected, "event": "broadcast"})
        wait_for_clients(selected, round_id)
        local_states = collect_client_updates(selected, round_id)
        contrib_info = read_contributions(selected, round_id)
        weights = compute_contribution_weights(selected, contrib_info)
        global_adapter = average_adapter_states_weighted(local_states, weights if weights else None)
        weight_map = {str(cid): weight for cid, weight in zip(selected, weights)} if weights else None
        _log_progress({
            "round": round_id,
            "phase": phase,
            "selected_clients": selected,
            "event": "aggregated",
            "contribution_weights": weight_map,
        })
        logger.info("Aggregated round %s phase %s with weights %s", round_id, phase, weight_map)
        round_id += 1
    return global_adapter, round_id


def run() -> None:
    """Entry point for the orchestrator container."""
    os.makedirs(EXCHANGE, exist_ok=True)
    logger.info("Federated training starting with %s clients.", NUM_CLIENTS)
    global_adapter = initial_adapter()
    global_adapter, next_round = run_phase(0, FED_ROUNDS, PHASE_SFT, global_adapter)
    _, model = get_peft_model_and_tokenizer()
    set_lora_state_dict(model, global_adapter)
    os.makedirs("logs", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("logs", "ref_model_sft.pth"))

    global_adapter, next_round = run_phase(next_round, FED_ROUNDS, PHASE_VA, global_adapter)
    torch.save(global_adapter, os.path.join("logs", "global_final.pth"))
    write_phase(next_round, PHASE_DONE)
    logger.info("Training complete. Final adapters saved.")


if __name__ == "__main__":
    run()
