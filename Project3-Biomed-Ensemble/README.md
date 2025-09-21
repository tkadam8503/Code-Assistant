# Project 3 — Ensemble Learning for Noisy Biomedical Image Classification (OCTMNIST)

We train **ResNet-50** and **DenseNet-121** on **OCTMNIST** (from **MedMNIST**), with **noise augmentations** to simulate low SNR. Then we do **confidence-weighted ensembling**.

## Dataset
- Uses `medmnist` package to download OCTMNIST automatically.

## Steps
```bash
pip install -r ../global-requirements.txt
pip install medmnist

# Train both backbones
python train_ensemble.py --backbone resnet50 --epochs 1 --out ./out_resnet
python train_ensemble.py --backbone densenet121 --epochs 1 --out ./out_densenet

# Evaluate individually and as an ensemble
python eval_ensemble.py --paths ./out_resnet ./out_densenet
```

## Files
- `train_ensemble.py` — trains a single backbone with noise aug.
- `eval_ensemble.py` — loads checkpoints, runs eval, and computes weighted ensemble.
