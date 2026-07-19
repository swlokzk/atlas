# Research Workflow

## Dataset

The target dataset is RadioML 2016.10a, normally supplied as:

```text
RML2016.10a_dict.pkl
```

The dataset inventory must record modulation classes, SNR levels, sample counts, checksum, split strategy, and random seed before deployment accuracy claims are made.

## Research Commands

Run commands from `_development/amr-gff-nn`:

```bash
python src/run.py evaluate --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl --batch-size 256
python src/run.py deep_analysis --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl
python src/run.py ablation --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl
python src/run.py gating --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl
python src/run.py compare --weights path/to/model.pth --data path/to/RML2016.10a_dict.pkl
```

The research workflow supports an explicit device option where implemented. CUDA use in research is allowed when the installed PyTorch build and hardware support it.

## Expected Analysis

- Overall accuracy.
- Per-class accuracy.
- Per-SNR accuracy.
- Count and normalized confusion matrices.
- Gating-weight distributions.
- Single-modality and pairwise ablation results.
- CNN versus Transformer comparisons.
- Optional t-SNE visualization.

## Separation From Deployment

Research scripts may require analysis and visualization dependencies that are not part of the minimal deployment environment. Do not use research output as deployment evidence without recording the dataset, split, seed, model configuration, and evaluation command.

Deployment preprocessing and inference must follow the contract documented in [Preprocessing Contract](04-preprocessing-contract.md), even when the research workflow uses a different data-loading path.
