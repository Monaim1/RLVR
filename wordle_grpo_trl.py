from __future__ import annotations

import random
import re
from typing import Any, List

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer


# Minimal system prompt and small 5-letter word list
GUESS_SYSTEM_PROMPT = (
    "You are a competitive game player.\n"
    "In each turn, output your guess inside <guess>...</guess> tags."
)

WORDS = [
    "about", "other", "which", "their", "there", "after", "first", "would", "these", "could",
    "right", "think", "great", "point", "small", "world", "place", "under", "light", "words",
]


def parse_guess(x: Any) -> tuple[str, float]:
    # Handle either plain strings or chat-like messages
    if isinstance(x, list) and all(isinstance(m, dict) for m in x):
        x = "\n".join(str(m.get("content", "")) for m in x)
    elif isinstance(x, dict) and "content" in x:
        x = str(x["content"])  # unlikely here
    else:
        x = str(x)

    m = list(re.finditer(r"<guess>\s*([a-zA-Z]+)\s*</guess>", x))
    if not m:
        return "", 0.0
    guess = m[-1].group(1).lower()
    return guess, 1.0


def make_dataset(n: int) -> Dataset:
    prompt = (
        f"{GUESS_SYSTEM_PROMPT}\n\n"
        "Guess the hidden 5-letter lowercase English word."
    )
    rows = [{"prompt": prompt, "answer": random.choice(WORDS)} for _ in range(n)]
    return Dataset.from_list(rows)


def reward_fn(prompts: List[Any], completions: List[Any], answer: List[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for comp, ans in zip(completions, answer):
        guess, fmt = parse_guess(comp)
        rewards.append(float((guess == ans)) + 0.1 * float(fmt))
    return rewards


def main():
    train_ds = make_dataset(500)
    eval_ds = make_dataset(100)

    config = GRPOConfig(
        output_dir="wordle-grpo",
        num_generations=4,
        max_prompt_length=128,
        max_completion_length=16,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        report_to=[],
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs=reward_fn,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()


if __name__ == "__main__":
    main()
