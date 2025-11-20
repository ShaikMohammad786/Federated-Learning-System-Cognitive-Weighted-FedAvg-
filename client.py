"""Federated client worker handling local training + FCCL contribution logs."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from typing import Dict

import torch

from training import load_client_data, local_dpo_train, local_sft_train
from utils import (
    device,
    extract_lora_state_dict,
    get_peft_model_and_tokenizer,
    load_tokenizer_and_base,
    set_lora_state_dict,
)

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)

# Default exchange directory shared with orchestrator inside Docker volumes.
EXCHANGE = os.environ.get("EXCHANGE_DIR", "/workspace/exchange")

# Phase labels and file patterns mirroring orchestrator.py protocol.
PHASE_SFT = "sft"
PHASE_VA = "va"
PHASE_DONE = "done"
PHASE_FILE_FMT = "phase_{round_id}.txt"
GLOBAL_FILE_FMT = "global_round_{round_id}.pt"
CLIENT_FILE_FMT = "client_{client_id}_round_{round_id}.pt"
DONE_FLAG_FMT = "done_{client_id}_{round_id}.flag"
CONTRIB_FILE_FMT = "contrib_client_{client_id}_round_{round_id}.json"
STATUS_FILE_FMT = "status_{client_id}.json"


def infer_client_id() -> int:
    """Infer client ID from $CLIENT_ID or trailing digits of hostname."""
    env_id = os.environ.get("CLIENT_ID")
    if env_id is not None:
        try:
            return int(env_id)
        except ValueError:
            return 0
    hostname = socket.gethostname()
    digits = ""
    for ch in reversed(hostname):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else 0


CLIENT_ID = infer_client_id()


def wait_for(path: str, timeout: int = 86400) -> None:
    """Block until a file exists (used for files like global_round_*.pt)."""
    start = time.time()
    while not os.path.exists(path):
        time.sleep(1)
        if time.time() - start > timeout:
            raise TimeoutError(f"Timeout waiting for {path}")


def write_status(state: str, round_id: int, phase: str) -> None:
    """Persist lightweight heartbeat JSON (`status_<id>.json`) for the dashboard."""
    payload = {
        "client": CLIENT_ID,
        "round": int(round_id),
        "phase": str(phase),
        "state": str(state),
        "ts": time.time(),
    }
    try:
        os.makedirs(EXCHANGE, exist_ok=True)
        with open(os.path.join(EXCHANGE, STATUS_FILE_FMT.format(client_id=CLIENT_ID)), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as exc:  # best-effort heartbeat
        logger.debug("Heartbeat write failed: %s", exc)


def _log_client_metric(phase: str, loss: float) -> None:
    os.makedirs("logs", exist_ok=True)
    record = {"client": CLIENT_ID, "phase": phase, "loss": loss}
    with open(os.path.join("logs", "client_metrics.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _write_contribution(round_id: int, phase: str, avg_loss: float, num_samples: int) -> None:
    """Store FCCL metadata (`contrib_client_*`) with avg_loss, data volume, and timestamp."""
    payload: Dict[str, object] = {
        "client": CLIENT_ID,
        "round": int(round_id),
        "phase": phase,
        "avg_loss": float(avg_loss) if avg_loss is not None else None,
        "num_samples": int(num_samples),
        "ts": time.time(),
    }
    contrib_path = os.path.join(EXCHANGE, CONTRIB_FILE_FMT.format(client_id=CLIENT_ID, round_id=round_id))
    with open(contrib_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def run_client() -> None:
    """Main client loop: react to `phase_*.txt`, train locally, upload adapters + contributions."""
    os.makedirs(EXCHANGE, exist_ok=True)
    round_id = 0
    while True:
        phase_path = os.path.join(EXCHANGE, PHASE_FILE_FMT.format(round_id=round_id))
        if not os.path.exists(phase_path):
            done_markers = []
            for fname in os.listdir(EXCHANGE):
                if not fname.startswith("phase_"):
                    continue
                try:
                    with open(os.path.join(EXCHANGE, fname), "r", encoding="utf-8") as f:
                        if f.read().strip() == PHASE_DONE:
                            done_markers.append(fname)
                except OSError as exc:
                    logger.debug("Unable to inspect %s: %s", fname, exc)
            if done_markers:
                write_status("stopped", round_id, "done")
                logger.info("Client %s exiting after done marker.", CLIENT_ID)
                break
            time.sleep(1)
            continue

        with open(phase_path, "r", encoding="utf-8") as f:
            phase = f.read().strip()
        if phase == PHASE_DONE:
            write_status("stopped", round_id, phase)
            logger.info("Client %s received done phase.", CLIENT_ID)
            break

        global_path = os.path.join(EXCHANGE, GLOBAL_FILE_FMT.format(round_id=round_id))
        wait_for(global_path)
        global_adapter = torch.load(global_path, map_location=device)

        tokenizer, model = get_peft_model_and_tokenizer()
        if global_adapter:
            set_lora_state_dict(model, global_adapter)
        sft_data, va_data = load_client_data(CLIENT_ID)

        write_status("training", round_id, phase)
        logger.info("Client %s training phase %s round %s", CLIENT_ID, phase, round_id)
        if phase == PHASE_SFT:
            model, avg_loss = local_sft_train(model, tokenizer, sft_data)
            num_samples = len(sft_data)
        elif phase == PHASE_VA:
            _, ref_model = load_tokenizer_and_base()
            ref_model.to(device).eval()
            model, avg_loss = local_dpo_train(model, tokenizer, va_data, ref_model)
            num_samples = len(va_data)
        else:
            raise ValueError(f"Unknown phase {phase}")
        _log_client_metric(phase, avg_loss)

        try:
            _write_contribution(round_id, phase, avg_loss, num_samples)
        except Exception as exc:
            logger.warning("Failed to write contribution JSON: %s", exc)

        adapter_state = extract_lora_state_dict(model)
        torch.save(adapter_state, os.path.join(EXCHANGE, CLIENT_FILE_FMT.format(client_id=CLIENT_ID, round_id=round_id)))
        done_path = os.path.join(EXCHANGE, DONE_FLAG_FMT.format(client_id=CLIENT_ID, round_id=round_id))
        with open(done_path, "w", encoding="utf-8") as flag_file:
            flag_file.write("")
        write_status("completed", round_id, phase)
        round_id += 1


if __name__ == "__main__":
    os.makedirs(EXCHANGE, exist_ok=True)
    logger.info("Client %s starting worker loop.", CLIENT_ID)
    run_client()
