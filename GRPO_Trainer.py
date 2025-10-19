

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from reward_ie import compute_reward, RELEVANT_FIELDS


def build_prompt(text: str) -> str:
    fields_str = ", ".join(RELEVANT_FIELDS)
    return (
        "Extract the following fields as JSON only (no extra text). "
        f"Fields: {{{fields_str}}}\n\n"
        f"DOCUMENT:\n{text}\n\n"
        "Return strictly a single JSON object with those keys."
    )


def df_to_trl_dataset(manifest_path: str, limit: int | None = None) -> Dataset:
    df = pd.read_parquet(manifest_path)
    if limit:
        df = df.head(limit)
    if "text" not in df.columns:
        raise ValueError("Manifest must contain a 'text' column. Re-run generateDatasets.py to pre-extract text.")

    # Load gold label content as strings to avoid file IO in reward loop
    gold_jsons: List[str] = []
    for p in df["gold_json_path"].tolist():
        with open(p, "r") as f:
            gold_jsons.append(f.read())

    prompts = [build_prompt(t) for t in df["text"].tolist()]

    hf_df = pd.DataFrame({
        "prompt": prompts,
        "gold_json": gold_jsons,
        "patent_id": df["patent_id"].tolist(),
    })
    return Dataset.from_pandas(hf_df, preserve_index=False)


def reward_ie_trl(prompts: List[Any], completions: List[Any], gold_json: List[str], **kwargs) -> List[float]:
    # TRL can pass conversational messages or plain text. Normalize to strings.
    def to_text(x: Any) -> str:
        # If chat messages: list of dicts with 'content'
        if isinstance(x, list) and all(isinstance(m, dict) for m in x):
            return "\n".join(str(m.get("content", "")) for m in x)
        if isinstance(x, dict) and "content" in x:
            return str(x["content"])  # unlikely here
        return str(x)

    outs: List[float] = []
    for comp, gold_s in zip(completions, gold_json):
        text = to_text(comp)
        gold = json.loads(gold_s)
        r, _ = compute_reward(text, gold)
        outs.append(float(r))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-manifest", default="Patent_Data/train_manifest.parquet")
    ap.add_argument("--val-manifest", default="Patent_Data/val_manifest.parquet")
    ap.add_argument("--model", default="Qwen/Qwen2-0.5B-Instruct")
    ap.add_argument("--output", default="Qwen2-0.5B-IE-GRPO")
    ap.add_argument("--train-limit", type=int, default=128)
    ap.add_argument("--val-limit", type=int, default=64)
    ap.add_argument("--num-generations", type=int, default=4)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--per-device-batch", type=int, default=1)
    ap.add_argument("--gradient-accumulation", type=int, default=4)
    args = ap.parse_args()

    train_ds = df_to_trl_dataset(args.train_manifest, limit=args.train_limit)
    val_ds = df_to_trl_dataset(args.val_manifest, limit=args.val_limit)

    training_args = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        bf16=False,
        fp16=False,  # MPS does not support fp16 training
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        logging_steps=10,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        report_to=[],  # set ["wandb"] if you want
        num_train_epochs=1.0,
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_ie_trl,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()


if __name__ == "__main__":
    main()

