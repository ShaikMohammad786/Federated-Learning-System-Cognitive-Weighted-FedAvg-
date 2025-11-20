"""Prepare non-IID client datasets for SFT (FedIT) and VA (FedVA)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Sequence

from datasets import load_dataset

from config import DATA_SFT_SPLIT, DATA_VA_SPLIT, NON_IID_ALPHA, NUM_CLIENTS

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)
random.seed(42)

os.makedirs("data/clients", exist_ok=True)


def non_iid_split(
    records: Sequence[Dict[str, str]],
    num_clients: int,
    alpha: float = NON_IID_ALPHA,
) -> List[List[Dict[str, str]]]:
    """Dirichlet-based non-IID splitter (alpha controls heterogeneity)."""
    if num_clients <= 0:
        return []
    if not records:
        return [[] for _ in range(num_clients)]
    shuffled = list(records)
    random.shuffle(shuffled)
    weights = [random.gammavariate(alpha, 1.0) for _ in range(num_clients)]
    total_weight = sum(weights)
    sizes = [max(0, int(len(shuffled) * (w / total_weight))) for w in weights]
    diff = len(shuffled) - sum(sizes)
    idx = 0
    while diff > 0:
        sizes[idx % num_clients] += 1
        diff -= 1
        idx += 1
    idx = 0
    while diff < 0:
        slot = idx % num_clients
        if sizes[slot] > 0:
            sizes[slot] -= 1
            diff += 1
        idx += 1

    if len(shuffled) >= num_clients:
        for i, size in enumerate(sizes):
            if size == 0:
                donor = max(range(num_clients), key=lambda j: sizes[j])
                if sizes[donor] > 1:
                    sizes[donor] -= 1
                    sizes[i] += 1

    splits: List[List[Dict[str, str]]] = []
    cursor = 0
    for size in sizes:
        splits.append(shuffled[cursor:cursor + size])
        cursor += size
    if cursor < len(shuffled):
        splits[-1].extend(shuffled[cursor:])
    return splits


def prepare_sft(
    split: str = DATA_SFT_SPLIT,
    num_clients: int = NUM_CLIENTS,
    alpha: float = NON_IID_ALPHA,
) -> None:
    """Download Alpaca data and partition across clients non-IID."""
    logger.info("Loading Alpaca slice %s ...", split)
    ds = load_dataset("tatsu-lab/alpaca", split=split)
    records: List[Dict[str, str]] = []
    for ex in ds:
        instr = ex.get("instruction", "")
        inp = ex.get("input", "")
        prompt = instr if not inp else instr + "\n" + inp
        resp = ex.get("output", "")
        if prompt and resp:
            records.append({"instruction": prompt, "response": resp})
    splits = non_iid_split(records, num_clients, alpha=alpha)
    total = 0
    for i, part in enumerate(splits):
        path = os.path.join("data", "clients", f"client_{i}_sft.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in part:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Client %s SFT samples: %s", i, len(part))
        total += len(part)
    logger.info("SFT prepared. total=%s slices to %s clients", total, num_clients)


def prepare_va(
    split: str = DATA_VA_SPLIT,
    num_clients: int = NUM_CLIENTS,
    alpha: float = NON_IID_ALPHA,
) -> None:
    """Download (or synthesize) VA preference data and split non-IID."""
    logger.info("Loading UltraFeedback preference dataset %s ...", split)
    records: List[Dict[str, str]] = []
    try:
        ds = load_dataset("openbmb/UltraFeedback", split=split)
        for ex in ds:
            completions = ex.get("completions", [])
            if not completions or len(completions) < 2:
                continue
            prompt = ex.get("instruction", "").strip()
            chosen = completions[0].get("response", "").strip()
            rejected = completions[1].get("response", "").strip()
            if prompt and chosen and rejected:
                records.append({"instruction": prompt, "preferred": chosen, "dispreferred": rejected})
    except Exception as exc:
        logger.warning("UltraFeedback load failed (%s). Falling back to synthetic VA.", exc)
        ds_sft = load_dataset("tatsu-lab/alpaca", split=DATA_SFT_SPLIT)
        sft_recs = []
        for ex in ds_sft:
            instr = ex.get("instruction", "")
            inp = ex.get("input", "")
            prompt = instr if not inp else instr + "\n" + inp
            resp = ex.get("output", "")
            if prompt and resp:
                sft_recs.append({"instruction": prompt, "response": resp})
        random.shuffle(sft_recs)
        for i, rec in enumerate(sft_recs):
            j = (i + 1) % len(sft_recs)
            records.append({
                "instruction": rec["instruction"],
                "preferred": rec["response"],
                "dispreferred": sft_recs[j]["response"],
            })

    logger.info("Filtered %s usable preference pairs", len(records))
    splits = non_iid_split(records, num_clients, alpha=alpha)
    total = 0
    for i, part in enumerate(splits):
        path = os.path.join("data", "clients", f"client_{i}_va.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in part:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Client %s VA samples: %s", i, len(part))
        total += len(part)

    logger.info("VA prepared. total=%s split to %s clients", total, num_clients)


def main() -> None:
    """CLI entrypoint for preparing SFT and VA datasets."""
    parser = argparse.ArgumentParser(description="Prepare non-IID client datasets for FedIT/FedVA.")
    parser.add_argument("--sft-split", default=DATA_SFT_SPLIT, help="Dataset slice for SFT (default: config.DATA_SFT_SPLIT).")
    parser.add_argument("--va-split", default=DATA_VA_SPLIT, help="Dataset slice for VA (default: config.DATA_VA_SPLIT).")
    parser.add_argument("--num-clients", type=int, default=NUM_CLIENTS, help="Number of federated clients.")
    parser.add_argument("--alpha", type=float, default=NON_IID_ALPHA, help="Dirichlet alpha (lower => more skew).")
    args = parser.parse_args()
    prepare_sft(split=args.sft_split, num_clients=args.num_clients, alpha=args.alpha)
    prepare_va(split=args.va_split, num_clients=args.num_clients, alpha=args.alpha)


if __name__ == "__main__":
    main()
