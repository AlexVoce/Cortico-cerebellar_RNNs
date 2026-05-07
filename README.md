# Cortico-cerebellar RNNs

This repository contains code for training and analysing recurrent neural networks augmented with a cerebellar-inspired feedforward bias module. The project tests whether modular cortico-cerebellar structure can improve learning efficiency on temporal sequencing tasks compared with recurrent-only baselines.

The core model is a recurrent network with an optional cerebellar-inspired module that receives the recurrent hidden state and the current task input. This module generates a hidden-sized bias signal that is injected back into the recurrent transition. Models are evaluated on curriculum-based temporal memory tasks, including delayed match-to-sample and parity.

## Project status
The manuscript is currently in preparation. Results, scripts, and documentation may be updated as the project is finalised.

The repository is intended for research reproducibility rather than as a general-purpose machine learning library. APIs may change as the manuscript and analyses are finalised.

---
## Repository structure

```text
.
├── model/
│   ├── models_cb.py              # Elman RNN and cerebellar bias module
│   └── GRU_test.py               # GRU variant with optional cerebellar bias (control)
│
├── tasks/
│   ├── task_registry.py          # Task specifications, losses, metrics, and advance rules
│   ├── tasks_using.py            # Sequence/task generation utilities
│   ├── multitask_impl.py         # Multi-task training implementation
│   ├── continual_impl.py         # Continual-learning implementation
│   └── task_switch_one.py        # Task-switching implementation
│
├── training/
│   ├── base.py                   # Shared train/evaluate utilities
│   ├── train_utils.py            # Gradient, optimizer, and module-freezing utilities
│   ├── train_reservoir.py        # Reservoir/interleaved-reservoir training
│   ├── variants.py               # Reservoir curriculum-stage logic
│   ├── train.py                  # Main training entry point
│   └── save.py                   # Checkpoint/result saving utilities
│
├── analysis/                     # Analysis and plotting scripts
│
├── results/                      # Results used for analysis
│   ├── GRU_test/
│   ├── multi_task/
│   ├── reservoir_comparisons/
│   ├── single_task/
│   └── task_switch/
│
├── final_analysis_figs.ipynb     # Figure generation + analysis notebook
└── README.md
```
