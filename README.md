# Atlas

Atlas is a curated monorepo that organizes research and development projects in one governed workspace.

## Overview

This repository focuses on two top-level domains:

- `_research/`: data science, machine learning, and analysis projects
- `_development/`: application, frontend, and engineering prototypes

Repository metadata and governance are maintained centrally via `catalog/` and `docs/`.

## Repository Structure

```text
atlas/
├── _research/
│   ├── cgb-anomaly-intervention-analysis/
│   ├── e-learning-fa/
│   └── machine-learning-applied-fmri/
├── _development/
│   ├── amr-gff-nn/
│   ├── binance-multi-assets-singal-agent/
│   └── r3f-portfolio/
├── catalog/
├── docs/
├── CONTRIBUTING.md
└── README.md
```

## Project Index

### Research Projects

- `cgb-anomaly-intervention-analysis`  
  Transformer-based anomaly/forecasting research workflows.
- `e-learning-fa`  
  E-learning data analysis pipeline and scripts.
- `machine-learning-applied-fmri`  
  Machine learning workflows for fMRI data.

### Development Projects

- `amr-gff-nn`  
  AMR Gated Fusion Former model project.
- `binance-multi-assets-singal-agent`  
  Multi-asset Binance signal agent prototype.
- `r3f-portfolio`  
  React + Three Fiber interactive portfolio app.

## Catalog and Governance

- Project catalog:
  - Human-readable: `catalog/index.md`
  - Machine-readable: `catalog/projects.yml`
- Governance docs:
  - `docs/ARCHITECTURE.md`
  - `docs/PROJECT_LIFECYCLE.md`
  - `docs/GOVERNANCE.md`
  - `docs/BRANCH_POLICY.md`

## Quick Start

### 1) Clone the repository

```bash
git clone https://github.com/sinwulok/atlas.git
cd atlas
```

### 2) Work on one subproject only (optional sparse-checkout)

```bash
git sparse-checkout init --cone
git sparse-checkout set _research/machine-learning-applied-fmri
```

### 3) Follow project-local setup

Each subproject has its own dependencies and run entrypoints:

- Python projects: check `requirements.txt` + `run.py`
- Frontend projects: check `package.json` scripts

## Contributing

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and keep changes scoped to the target project.

## Notes

- The previous repository overview is archived as `README_ARC_V2`.
- Some metadata files may still contain historical aliases and can be normalized in a follow-up pass.
