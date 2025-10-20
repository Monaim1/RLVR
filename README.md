# RLVR

Minimal pipeline for document information extraction using supervised fine‑tuning (SFT) and GRPO reinforcement learning on Hugging Face Transformers/TRL. The example task targets patent PDFs with a JSON reward based on key field accuracy and simple constraints.

## Setup
- Python 3.13 (`.python-version` provided)
- Create a virtual env, then install deps:
  - Recommended: `pip install -e .` (uses `pyproject.toml`)
  - Or: `pip install -r requirements.txt`

## Data
- Place raw patent JSON files under `Patent_Data/Raw_data`.
- Generate PDFs, labels, and manifests (with pre‑extracted text):
  - `python generateDatasets.py`
  - Outputs: `Patent_Data/train_manifest.parquet`, `Patent_Data/val_manifest.parquet`

## Train
- Optional SFT warm‑up:
  - `python SFT_warmUP.py --model Qwen/Qwen2-0.5B-Instruct`
- GRPO training with TRL:
  - `python GRPO_Trainer.py --model Qwen/Qwen2-0.5B-Instruct`
  - Uses manifests in `Patent_Data/` and the reward in `reward_IE.py`
- Alternative loop example (requires Unsloth):
  - `python grpo_loop.py`

## Notebooks
- `IE_GRPO.ipynb`, `Qwen3_(4B)_GRPO.ipynb` show end‑to‑end runs and experimentation.

## Notes
- Key scripts: `generateDatasets.py`, `SFT_warmUP.py`, `GRPO_Trainer.py`, `grpo_loop.py`, `reward_IE.py`, `patent_dataset.py`.
- Set `MODEL_NAME` env var to override defaults where supported.
