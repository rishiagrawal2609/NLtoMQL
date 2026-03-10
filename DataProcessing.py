"""
Advanced NL-to-MQL trainer using HuggingFace Trainer class.

Base model: SmolLM3-3B (HuggingFace, 2025) — state-of-the-art 3B SLM.
Accelerator: Apple Silicon MPS → NVIDIA CUDA → CPU (auto-detected).

Features over NLtoMQL_SLM.py:
  - HF Trainer with gradient accumulation, checkpointing, logging
  - Proper dataset tokenization via datasets library
  - DataCollator for dynamic padding
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


DEFAULT_CSV = "atlas_sample_data_benchmark.csv"
DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM3-3B"


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

# Data loading
def load_training_pairs(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    required_columns = ["input.nlQuery", "expected.dbQuery"]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    data = data[required_columns].dropna().drop_duplicates().reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid training rows found in the CSV.")
    return data

# Prompt engineering
def to_prompt(nl_query: str) -> str:
    sys_msg = (
        "You are an expert MongoDB query generator. "
        "You ONLY output valid MongoDB MQL. "
        "Do not explain your code. Do not use markdown blocks. "
        "Given the user's schema and request, generate the MQL."
    )
    return (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{nl_query.strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def to_training_text(nl_query: str, mql_query: str) -> str:
    return f"{to_prompt(nl_query)}{mql_query.strip()}<|im_end|>"

# Dataset builder
@dataclass
class NLToMQLDatasetBuilder:
    tokenizer: AutoTokenizer
    max_length: int

    def build(self, dataframe: pd.DataFrame) -> Dataset:
        dataset = Dataset.from_pandas(
            dataframe.rename(
                columns={
                    "input.nlQuery": "nl_query",
                    "expected.dbQuery": "mql_query",
                }
            )
        )

        def tokenize_batch(batch):
            texts = [
                to_training_text(nl, mql)
                for nl, mql in zip(batch["nl_query"], batch["mql_query"])
            ]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(tokenize_batch, batched=True)
        return tokenized_dataset.remove_columns(
            [column for column in tokenized_dataset.column_names if column not in {"input_ids", "attention_mask", "labels"}]
        )

# Training
def train_model(
    csv_path: str,
    base_model_name: str,
    output_dir: str,
    num_train_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
) -> None:
    device, dtype = get_device_and_dtype()
    print(f"Training on device: {device} with dtype: {dtype}")

    dataframe = load_training_pairs(csv_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model — device_map="auto" only for CUDA; MPS/CPU need manual .to()
    if device.startswith("cuda"):
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
        )
        # For MPS, we don't call .to() here — Trainer handles placement

    # Detect target modules for LoRA
    if any("q_proj" in name for name, _ in model.named_modules()):
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = ["c_attn"]

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    builder = NLToMQLDatasetBuilder(tokenizer=tokenizer, max_length=max_length)
    train_dataset = builder.build(dataframe)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Build TrainingArguments with MPS-aware settings
    use_fp16 = device.startswith("cuda")
    # MPS supports bf16 on newer PyTorch but float32 is safest
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=use_fp16,
        bf16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Apple Silicon MPS: use the MPS device automatically via PyTorch
        use_mps_device=device == "mps",
        dataloader_pin_memory=device.startswith("cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    metadata = {
        "base_model": base_model_name,
        "csv_path": csv_path,
        "rows_used": len(dataframe),
        "max_length": max_length,
        "device_used": device,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"✓ Training completed. Adapter + tokenizer saved to: {output_dir}")

# Inference
def generate_mql(
    adapter_dir: str,
    nl_query: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    metadata_path = Path(adapter_dir) / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    base_model_name = metadata["base_model"]

    device, dtype = get_device_and_dtype()
    print(f"Running inference on {device} ({dtype})...")

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.startswith("cuda"):
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
        )
        base_model = base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    prompt = to_prompt(nl_query)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).strip()
    
    for stop_word in ["<|im_end|>", "User request:", "MQL:", "<|im_start|>"]:
        if stop_word in generated_text:
            generated_text = generated_text.split(stop_word)[0].strip()
            
    return generated_text


# CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run a state-of-the-art NL-to-MQL model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Fine-tune an SLM adapter on NL→MQL pairs.")
    train_parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to training CSV file.")
    train_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="HF base model name.")
    train_parser.add_argument("--output-dir", default="models/nl2mql-lora", help="Where to save the adapter.")
    train_parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    train_parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    train_parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    train_parser.add_argument("--max-length", type=int, default=512, help="Max token length.")

    infer_parser = subparsers.add_parser("infer", help="Generate MQL from a natural language query.")
    infer_parser.add_argument("--adapter-dir", default="models/nl2mql-lora", help="Path to trained adapter.")
    infer_parser.add_argument("--nl-query", required=True, help="Natural language request.")
    infer_parser.add_argument("--max-new-tokens", type=int, default=220, help="Generation length.")
    infer_parser.add_argument("--temperature", type=float, default=0.0, help="Set 0 for deterministic decoding.")
    infer_parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(
            csv_path=args.csv,
            base_model_name=args.base_model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
        )
    elif args.command == "infer":
        result = generate_mql(
            adapter_dir=args.adapter_dir,
            nl_query=args.nl_query,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(result)


if __name__ == "__main__":
    main()
