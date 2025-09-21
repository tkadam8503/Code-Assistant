# LLM Code Assistant

A minimal, runnable **Jupyter** project that generates Python code with a small LLM and includes an **auto‑debug** loop. Optional: extend to LoRA fine‑tuning on the MBPP dataset.

## Quickstart
```bash
# create env (optional)
conda create -n code-assistant python=3.10 -y
conda activate code-assistant

# install
pip install -r requirements.txt
```

Then open **Project1_CodeAssistant.ipynb** and run cells from top to bottom.

## What’s inside
- `Project1_CodeAssistant.ipynb` — generation + auto‑debug demo (runnable on CPU).
- `requirements.txt` — lightweight dependencies for the notebook.

## Notes
- Default base model is `gpt2` (small, quick to run). Swap to a tiny code model if you have GPU memory.
- This repo is intentionally simple and self‑contained for portfolio use.
