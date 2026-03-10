#!/usr/bin/env python3
"""
State-of-the-art SLM for NL-to-MQL generation using LoRA fine-tuning.

Base model: SmolLM3-3B (HuggingFace, 2025) — outperforms Llama-3.2-3B
             and Qwen2.5-3B on reasoning benchmarks.
Accelerator: Apple Silicon MPS → NVIDIA CUDA → CPU (auto-detected).

Supports two commands:
  1. train    - Fine-tune a LoRA adapter on CSV pairs of (nlQuery, mqlQuery)
  2. infer    - Generate MQL from a natural language text prompt

Example usage:
  # Train an adapter (will download base model ~6GB on first run):
  .venv/bin/python NLtoMQL_SLM.py train --csv atlas_sample_data_benchmark.csv

  # Infer on a test query (uses the trained adapter):
  .venv/bin/python NLtoMQL_SLM.py infer --nl "Find all documents where age > 25"
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Defaults
DEFAULT_CSV = "atlas_sample_data_benchmark.csv"
DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"
DEFAULT_ADAPTER_DIR = "models/nl2mql-lora"


# Device helpers — Apple Silicon MPS first, then CUDA, then CPU
def get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Pick the best available accelerator and matching dtype.

    MPS (Apple Silicon):  float32 — MPS doesn't fully support float16 for
                          many training/backward ops yet.
    CUDA:                 float16 — standard mixed-precision path.
    CPU:                  float32 — safe fallback.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps", torch.float32
    if torch.cuda.is_available():
        return "cuda:0", torch.float16
    return "cpu", torch.float32


def _move_to_device(tensors: dict, device: str) -> dict:
    """Move a dict of tensors to the target device."""
    return {k: v.to(device) for k, v in tensors.items()}


# Data loading
def load_training_pairs(csv_path: str) -> pd.DataFrame:
    """Load and validate CSV with nlQuery and mqlQuery columns."""
    df = pd.read_csv(csv_path)
    required = ["input.nlQuery", "expected.dbQuery"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[required].dropna().drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid training pairs in CSV.")
    logger.info(f"Loaded {len(df)} training pairs from {csv_path}")
    return df


# Prompt engineering
def create_prompt(nl_query: str) -> str:
    """Create a robust, chat-formatted prompt for the model."""
    sys_msg = (
        "You are an expert MongoDB query generator. "
        "You ONLY output valid MongoDB MQL queries. "
        "Do not explain your code. Do not wrap in markdown blocks. "
        "Given the user's schema and request, generate the MQL."
    )
    
    return (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{nl_query.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# LoRA target-module detection
def get_target_modules(model) -> list[str]:
    """Detect attention projection names that exist in the model."""
    module_names = {name for name, _ in model.named_modules()}
    # SmolLM3 / LLaMA-family use q_proj + v_proj
    candidates = [["q_proj", "v_proj"], ["c_attn"], ["out_proj"]]
    for group in candidates:
        if all(any(t in n for n in module_names) for t in group):
            return group
    return ["q_proj", "v_proj"]          # sensible default


# Training
def train_adapter(
    csv_path: str,
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = DEFAULT_ADAPTER_DIR,
    learning_rate: float = 3e-4,
    batch_size: int = 1,
    epochs: int = 3,
) -> None:
    """Fine-tune a LoRA adapter on the provided CSV pairs."""
    device, dtype = get_device_and_dtype()
    logger.info(f"Starting training with {base_model} on {device} ({dtype})...")

    # Load data
    df = load_training_pairs(csv_path)

    # Load base model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" works for CUDA; for MPS/CPU we load then move
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
        )
        model = model.to(device)

    # Setup LoRA — higher rank for the larger 3B model
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=get_target_modules(model),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for idx, row in df.iterrows():
            nl_query = row["input.nlQuery"]
            mql_query = row["expected.dbQuery"]

            text = f"{create_prompt(nl_query)} {mql_query.strip()}"
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )

            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)
            labels = input_ids.clone()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability (especially important on MPS)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.detach().item()

            if (idx + 1) % max(1, len(df) // 5) == 0:
                avg = total_loss / (idx + 1)
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Step {idx+1}/{len(df)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Avg: {avg:.4f}"
                )

        epoch_avg = total_loss / len(df)
        logger.info(f"Epoch {epoch+1}/{epochs} complete — avg loss: {epoch_avg:.4f}")

    # Save adapter
    adapter_path = Path(output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(adapter_path / "adapter_model")
    tokenizer.save_pretrained(adapter_path / "tokenizer")

    metadata = {
        "base_model": base_model,
        "rows_trained": len(df),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "device_used": device,
    }
    (adapter_path / "config.json").write_text(json.dumps(metadata, indent=2))

    logger.info(f"✓ Training complete! Adapter saved to {output_dir}")


# Inference
def generate_mql(
    nl_query: str,
    adapter_dir: str = DEFAULT_ADAPTER_DIR,
    max_new_tokens: int = 1000,
    temperature: float = 0.0,
) -> str:
    """Generate MQL from a natural language query using the trained adapter."""
    config_path = Path(adapter_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Adapter config not found at {adapter_dir}")

    config = json.loads(config_path.read_text())
    base_model = config["base_model"]

    device, dtype = get_device_and_dtype()
    logger.info(f"Running inference on {device} ({dtype})...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir + "/tokenizer", use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    if device.startswith("cuda"):
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
        )
        base = base.to(device)

    model = PeftModel.from_pretrained(base, adapter_dir + "/adapter_model")
    model.eval()

    # Generate
    prompt = create_prompt(nl_query)
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(output[0][len(tokens[0]):], skip_special_tokens=True).strip()
    
    # Prune any trailing chat tokens or hallucinations
    for stop_word in ["<|im_end|>", "User request:", "MQL:", "<|im_start|>"]:
        if stop_word in result:
            result = result.split(stop_word)[0].strip()
            
    return result


# CLI
def main():
    parser = argparse.ArgumentParser(description="State-of-the-art SLM for NL-to-MQL generation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train subcommand
    train_cmd = subparsers.add_parser("train", help="Train LoRA adapter on CSV")
    train_cmd.add_argument("--csv", default=DEFAULT_CSV, help="Training CSV path")
    train_cmd.add_argument("--model", default=DEFAULT_BASE_MODEL, help="Base model name")
    train_cmd.add_argument("--output", default=DEFAULT_ADAPTER_DIR, help="Output adapter dir")
    train_cmd.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    train_cmd.add_argument("--epochs", type=int, default=3, help="Training epochs")

    # Infer subcommand
    infer_cmd = subparsers.add_parser("infer", help="Generate MQL from NL")
    infer_cmd.add_argument("--nl", required=True, help="Natural language query")
    infer_cmd.add_argument("--adapter", default=DEFAULT_ADAPTER_DIR, help="Adapter dir")
    infer_cmd.add_argument("--temp", type=float, default=0.2, help="Temperature")
    infer_cmd.add_argument("--max-tokens", type=int, default=1000, help="Max new tokens")

    args = parser.parse_args()

    if args.command == "train":
        train_adapter(
            csv_path=args.csv,
            base_model=args.model,
            output_dir=args.output,
            learning_rate=args.lr,
            epochs=args.epochs,
        )
    elif args.command == "infer":
        mql = generate_mql(
            nl_query=args.nl,
            adapter_dir=args.adapter,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
        )
        print(mql)


if __name__ == "__main__":
    main()
