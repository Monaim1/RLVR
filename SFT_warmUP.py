from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig

from reward_ie import RELEVANT_FIELDS


def build_prompt(text: str) -> str:
    fields_str = ", ".join(RELEVANT_FIELDS)
    return (
        "Extract the following fields as JSON only (no extra text). "
        f"Fields: {{{fields_str}}}\n\n"
        f"DOCUMENT:\n{text}\n\n"
        "Return strictly a single JSON object with those keys."
    )


def df_to_sft_dataset(manifest_path: str, limit: int | None = None) -> Dataset:
    df = pd.read_parquet(manifest_path)
    if limit:
        df = df.head(limit)
    if "text" not in df.columns:
        raise ValueError("Manifest must contain a 'text' column. Re-run generateDatasets.py to pre-extract text.")

    prompts: List[str] = [build_prompt(t) for t in df["text"].tolist()]

    # Load gold JSON content as the target response
    answers: List[str] = []
    for p in df["gold_json_path"].tolist():
        with open(p, "r") as f:
            answers.append(f.read())

    # Single text field: prompt + answer delimited
    samples = [
        {
            "text": f"{prompt}\n\n<answer>\n{answer}\n</answer>",
        }
        for prompt, answer in zip(prompts, answers)
    ]
    return Dataset.from_list(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", default="Patent_Data/train_manifest.parquet")
    ap.add_argument("--val-manifest", default="Patent_Data/val_manifest.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--output", default="Qwen2-0.5B-IE-SFT")
    ap.add_argument("--train-limit", type=int, default=200)
    ap.add_argument("--val-limit", type=int, default=100)
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--gradient-accumulation", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    args = ap.parse_args()

    train_ds = df_to_sft_dataset(args.train_manifest, limit=args.train_limit)
    val_ds = df_to_sft_dataset(args.val_manifest, limit=args.val_limit)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sft_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        fp16=False,
        bf16=False,
        report_to=[],  # set ["wandb"] if you want logging
    )

    trainer = SFTTrainer(
        model=args.model,
        tokenizer=tok,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )

    trainer.train()


if __name__ == "__main__":
    main()

