"""Evaluate base vs FCCL-tuned GPT-2 using perplexity, preferences, and samples."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List, Sequence, Tuple

import torch
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from config import LORA_ALPHA, LORA_R, MAX_LENGTH, MODEL_NAME
from utils import device, read_jsonl, seq_logprob

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger(__name__)


def load_base_model() -> Tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]:
    """Load the base GPT-2 model for comparison."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model


def load_finetuned_model(model_path: str = "logs/global_final.pth") -> Tuple[PreTrainedTokenizerBase, AutoModelForCausalLM]:
    """Load the global LoRA-adapted model produced by federated training."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file {model_path} not found. Run orchestrator/client training before evaluation."
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["c_attn"],
        inference_mode=True,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    return tokenizer, model


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
) -> float:
    """Compute perplexity over a list of instruction-response strings."""
    total_loss = 0.0
    total_tokens = 0
    for text in tqdm(texts, desc="Computing perplexity"):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            total_loss += outputs.loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    return torch.exp(torch.tensor(avg_loss)).item()


def compute_preference_accuracy(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    va_data: Sequence[Dict[str, str]],
) -> float:
    """Measure % of VA pairs where the model prefers the reference response."""
    correct = 0
    total = 0
    for ex in tqdm(va_data, desc="Computing preference accuracy"):
        instr = ex["instruction"]
        pref = ex["preferred"]
        disp = ex["dispreferred"]
        seq_pref = f"Instruction: {instr}\nResponse: {pref}"
        seq_disp = f"Instruction: {instr}\nResponse: {disp}"
        lp_pref = seq_logprob(model, tokenizer, seq_pref)
        lp_disp = seq_logprob(model, tokenizer, seq_disp)
        if lp_pref > lp_disp:
            correct += 1
        total += 1
    return correct / total if total else 0.0


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_new_tokens: int = 50,
) -> str:
    """Generate a response continuation for logging qualitative samples."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if full_text.startswith("Instruction:"):
        full_text = full_text.split("Response:", 1)[-1].strip() if "Response:" in full_text else full_text
    return full_text


def gather_local_test_sets(
    max_sft_pairs: int = 50,
    max_va_pairs: int = 50,
) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
    """Collect small eval slices from client data dumps."""
    sft_pairs: List[Tuple[str, str]] = []
    va_pairs: List[Dict[str, str]] = []
    data_dir = os.path.join("data", "clients")
    if not os.path.isdir(data_dir):
        return sft_pairs, va_pairs
    for fname in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fname)
        if fname.endswith("_sft.jsonl"):
            for ex in read_jsonl(path):
                instr = ex.get("instruction", "").strip()
                resp = ex.get("response", "").strip()
                if instr and resp:
                    sft_pairs.append((instr, resp))
                    if len(sft_pairs) >= max_sft_pairs:
                        break
        elif fname.endswith("_va.jsonl"):
            for ex in read_jsonl(path):
                instr = ex.get("instruction", "").strip()
                pref = ex.get("preferred", "").strip()
                disp = ex.get("dispreferred", "").strip()
                if instr and pref and disp:
                    va_pairs.append({
                        "instruction": instr,
                        "preferred": pref,
                        "dispreferred": disp,
                    })
                    if len(va_pairs) >= max_va_pairs:
                        break
    return sft_pairs, va_pairs


def evaluate_models(
    max_sft_pairs: int = 50,
    max_va_pairs: int = 50,
    skip_base: bool = False,
    gen_samples: int = 3,
    gen_tokens: int = 30,
) -> None:
    """End-to-end evaluation comparing base and FCCL-tuned models."""
    base_tokenizer = base_model = None
    if not skip_base:
        logger.info("Loading base model %s ...", MODEL_NAME)
        base_tokenizer, base_model = load_base_model()
    logger.info("Loading fine-tuned model from logs/global_final.pth ...")
    finetuned_tokenizer, finetuned_model = load_finetuned_model()

    logger.info("Loading local test data from data/clients")
    sft_pairs, va_pairs = gather_local_test_sets(max_sft_pairs=max_sft_pairs, max_va_pairs=max_va_pairs)
    sft_texts = [f"Instruction: {p}\nResponse: {r}" for (p, r) in sft_pairs]

    logger.info("=== Perplexity Evaluation ===")
    if not skip_base and base_model and base_tokenizer:
        base_perplexity = compute_perplexity(base_model, base_tokenizer, sft_texts)
        logger.info("Base model perplexity: %.2f", base_perplexity)
    finetuned_perplexity = compute_perplexity(finetuned_model, finetuned_tokenizer, sft_texts)
    logger.info("Fine-tuned model perplexity: %.2f", finetuned_perplexity)

    logger.info("=== Preference Accuracy Evaluation ===")
    if not skip_base and base_model and base_tokenizer:
        base_pref_accuracy = compute_preference_accuracy(base_model, base_tokenizer, va_pairs)
        logger.info("Base model preference accuracy: %.2f%%", base_pref_accuracy * 100)
    finetuned_pref_accuracy = compute_preference_accuracy(finetuned_model, finetuned_tokenizer, va_pairs)
    logger.info("Fine-tuned model preference accuracy: %.2f%%", finetuned_pref_accuracy * 100)

    logger.info("=== Sample Response Evaluation ===")
    for i, (prompt, reference) in enumerate(sft_pairs[:gen_samples], 1):
        logger.info("Sample %s prompt: %s", i, prompt)
        if not skip_base and base_model and base_tokenizer:
            base_generated = generate_response(
                base_model,
                base_tokenizer,
                f"Instruction: {prompt}\nResponse:",
                max_new_tokens=gen_tokens,
            )
            logger.info("Base model response: %s", base_generated)
        finetuned_generated = generate_response(
            finetuned_model,
            finetuned_tokenizer,
            f"Instruction: {prompt}\nResponse:",
            max_new_tokens=gen_tokens,
        )
        logger.info("Fine-tuned model response: %s", finetuned_generated)
        logger.info("Reference response: %s", reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base vs FCCL-tuned GPT-2 adapters.")
    parser.add_argument("--sft", type=int, default=50, help="Max instruction/response pairs for perplexity.")
    parser.add_argument("--va", type=int, default=50, help="Max preference pairs for accuracy.")
    parser.add_argument("--skip-base", action="store_true", help="Skip base metrics for faster runs.")
    parser.add_argument("--gen-samples", type=int, default=3, help="Number of qualitative samples to log.")
    parser.add_argument("--gen-tokens", type=int, default=30, help="Max new tokens per generation.")
    args = parser.parse_args()
    evaluate_models(
        max_sft_pairs=args.sft,
        max_va_pairs=args.va,
        skip_base=args.skip_base,
        gen_samples=args.gen_samples,
        gen_tokens=args.gen_tokens,
    )