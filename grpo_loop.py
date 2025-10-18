"""
Sketch of a GRPO training loop with Unsloth/TRL for RLVR-style IE.

This file wires:
 - PatentIEDataset (from patent_dataset.py)
 - Reward function (from reward_ie.py)
 - A group sampling loop (K generations per prompt)
 - Group-wise reward normalization and GRPO trainer step

Note: Adjust imports per your Unsloth version. This is an illustrative
implementation you can adapt to your setup.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from patent_dataset import PatentIEDataset, load_manifest
from reward_ie import compute_reward, batch_compute_rewards
from unsloth import GRPOTrainer  



def format_batch(batch: Dict[str, Any]) -> List[str]:
    return batch["input_text"]


def grpo_env_generate(
    model, tokenizer, prompts: List[str], K: int = 4,
    max_new_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9
) -> List[List[str]]:
    device = getattr(model, "device", torch.device("cpu"))
    outputs: List[List[str]] = []
    for p in prompts:
        gens = []
        inputs = tokenizer(p, return_tensors="pt").to(device)
        for _ in range(K):
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(gen[0], skip_special_tokens=True)
            gens.append(text)
        outputs.append(gens)
    return outputs


def normalize_group_rewards(group_rewards: List[List[float]]) -> List[List[float]]:
    normed: List[List[float]] = []
    for rs in group_rewards:
        if not rs:
            normed.append(rs)
            continue
        t = torch.tensor(rs, dtype=torch.float32)
        mu, std = t.mean(), t.std(unbiased=False)
        if std.item() == 0:
            norm = (t - mu)
        else:
            norm = (t - mu) / (std + 1e-6)
        normed.append(norm.tolist())
    return normed


def build_dataloader(manifest_path: str, batch_size: int = 2, preload_text: bool = True) -> DataLoader:
    df = load_manifest(manifest_path)
    ds = PatentIEDataset(df, preload_text=preload_text)

    def collate(batch_list: List[Dict[str, Any]]):
        return {
            "input_text": [b["input_text"] for b in batch_list],
            "gold": [b["gold"] for b in batch_list],
            "patent_id": [b["patent_id"] for b in batch_list],
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)


def main():
    # Model setup (adjust model name / bits / LoRA per your infra)
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    group_size = int(os.environ.get("GROUP_SIZE", 4))
    train_manifest = os.environ.get("TRAIN_MANIFEST", "Patent_Data/train_manifest.parquet")
    train_loader = build_dataloader(train_manifest, batch_size=2, preload_text=True)

    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, group_size=group_size)

    # One illustrative pass
    for step_idx, batch in enumerate(train_loader):
        prompts = format_batch(batch)
        gens = grpo_env_generate(model, tokenizer, prompts, K=group_size)

        # Rewards
        golds = batch["gold"]
        raw_rewards = batch_compute_rewards(gens, golds)
        norm_rewards = [normalize_group_rewards([rs])[0] for rs in raw_rewards]

        # If Unsloth available, take a GRPO step
        if trainer is not None:
            trainer.step(prompts, gens, norm_rewards)

        # Minimal logging
        if step_idx % 5 == 0:
            print(f"Step {step_idx}: mean reward {[sum(rs)/len(rs) if rs else 0 for rs in raw_rewards]}")
        if step_idx >= 0:  # run a single batch by default
            break


if __name__ == "__main__":
    main()

