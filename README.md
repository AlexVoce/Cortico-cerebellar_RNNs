# Cortico-cerebellar RNNs

Code and results for training and analysing recurrent neural networks with a cerebellar-inspired bias module (CB-RNNs).

The model is a recurrent network with an optional cerebellar-inspired feedforward module. The module reads the recurrent hidden state and the current input, expands them into a large granule-cell-like layer, and returns a hidden-sized bias that is added to the recurrent update. The networks are trained with a curriculum on temporal memory tasks and their curriculum trajectories compared with recurrent-only baselines. 

The trained runs used in the paper are included under `results/`, so every figure and table can be reproduced without retraining.

## Setup

Tested with Python 3.10.

```bash
pip install -r requirements.txt
```

The figures use LaTeX text rendering (`text.usetex`), so a LaTeX installation is needed. Cell 2 of the notebook adds the macOS TeX path to `PATH`; on other systems it has no effect and can be removed.

## Reproducing the figures

Run `paper_figures.ipynb` from the repository root. It is organised by figure (Figures 2-6, then the appendix figures and tables) and reads everything from `results/`.

## Training

`training/train.py` is the single entry point for all training runs. Each run folder in `results/` contains a `config.json` that records the arguments it was trained with (`cli_args`). For example, one single-task DMS CB-RNN:

```bash
python training/train.py --task dms --model_type elman --curriculum_type single \
    --num_neurons 64 --gc_dim 256 --cb_sees_input --readout_mode single --afunc leakyrelu \
    --rnn_lr 0.01 --cb_lr 0.01 --num_epochs 1000 \
    --base_path ./results/my_runs
```

Add `--no_cb` for a recurrent-only baseline. `--multitask`, `--ct_switch` and `--curriculum_type reservoir` select the multitask, task-switching and reservoir training modes. `python training/train.py --help` lists every option.

## Repository structure

```text
.
├── paper_figures.ipynb           # All paper figures and tables
├── model/
│   ├── cb_rnn.py                 # Elman RNN with the cerebellar bias module
│   └── cb_gru.py                 # GRU variant (control)
├── tasks/
│   ├── task_generators.py        # Sequence generation for each task
│   ├── registry.py               # Task specifications, losses, metrics and curriculum rules
│   ├── multitask.py              # Multitask training
│   ├── task_switch.py            # Task-switching training
│   └── continual.py              # Sequential (continual) training
├── training/
│   ├── train.py                  # Training entry point
│   ├── base.py                   # Shared train/evaluate loop
│   ├── reservoir.py              # Reservoir training
│   ├── variants.py               # Reservoir curriculum stages
│   ├── utils.py                  # Gradient, optimiser and freezing utilities
│   └── save.py                   # Checkpoints, configs and stats
├── analysis/
│   ├── single_task_*.py          # Learning curves and summaries (Fig 2, App. D-E)
│   ├── scaling.py                # Parameter scaling (Fig 2)
│   ├── multitask_switching_plotting_utils.py   # Multitask and task switching (Figs 3-4)
│   ├── learning_metrics.py       # Learning-speed tables (App. C, K)
│   ├── cb_ablation.py            # CB ablation and class separation (Fig 5, App. I)
│   ├── readout_recovery.py       # Readout retraining without the CB (App. I)
│   ├── pca_dimensionality.py     # Hidden-state dimensionality (App. H)
│   ├── timescales.py             # Timescales (App. G); data from run_cb_timescales.py
│   ├── mechanism_plots.py        # Figure 6
│   └── ...                       # Figure 6 data pipelines (dms_*, parity_*, jacobian, model_stepping)
└── results/
    ├── single_task/              # DMS and parity, CB-RNN and RNN at several sizes
    ├── reservoir_comparisons/    # Reservoir-trained CB-RNNs
    ├── multi_task/
    ├── task_switch/
    ├── GRU_test/                 # GRU controls
    └── mechanistic_analysis/     # Figure 6 data
```
