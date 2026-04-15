# Cross-Domain QCL — Experiment Framework

This repository contains the code and paper for **"Cross-Domain Pre-Training to Mitigate Catastrophic Forgetting in Sequential Variational Quantum Classifiers"**, submitted to QUANTICS 2026 (Porto).

The framework was designed to be reusable for other quantum machine learning papers. If you swap `config.yaml` and the task-specific code in `data/` and `circuits/`, you can run a completely different experiment on Hercules without touching the orchestration layer.

---

## What this paper does

When you train a variational quantum circuit (VQC) on two tasks in sequence — Task A, then Task B — the circuit forgets Task A. This is **catastrophic forgetting**, and it is a known problem in continual learning. Classical solutions like replay buffers or elastic weight consolidation do not map cleanly onto quantum circuits.

This paper investigates a simpler idea: **pre-train the circuit on a synthetic Gaussian dataset** before running the sequential tasks. The synthetic domain has no overlap with the real image tasks (Fashion-MNIST and MNIST), so it acts as a neutral warm-up that places parameters in a stable region of the optimization landscape. The result is a lower forgetting drop on Task A after Task B training, without any memory buffer.

Three circuit architectures are tested:
- **Strongly Entangling Layers (SEL)** — high expressibility, many parameters
- **Basic Entangler** — minimal parametrization, nearest-neighbor CNOT chain
- **Tree Tensor Network (TTN)** — hierarchical structure, avoids global entanglement

All three are evaluated under both ideal simulation and a realistic noise model calibrated from IBM Heron r2 hardware data (T1 = 250 µs, T2 = 150 µs, gate errors from April 2025 calibration). Experiments run on the Hercules HPC cluster (CICA, Seville) over 5 random seeds.

---

## Repository structure

```
cross-domain-qcl/
├── circuits/
│   ├── ansatz.py          # SEL, BasicEntangler, TTN as PennyLane QNodes
│   └── noise.py           # IBM Heron r2 noise model (amplitude/phase damping + depolarizing)
├── data/
│   ├── loader.py          # Fashion-MNIST and MNIST loaders with PCA reduction
│   └── synthetic.py       # Synthetic Gaussian dataset generator
├── paper/
│   ├── main.tex           # Full paper (LLNCS format)
│   ├── references.bib     # Bibliography
│   └── tables/            # Auto-generated LaTeX tables (from generate_tables.py)
├── trainer.py             # QCL training loop: pre-train → Task A → Task B → forgetting
├── runner.py              # Experiment orchestrator (generates SLURM command files)
├── manager.py             # Interactive HUB for Hercules (reads phases from config.yaml)
├── generate_tables.py     # Generates LaTeX result tables from results/
├── slurm_generic.sh       # SLURM array template (Hercules-compatible)
├── deploy.sh              # rsync script to sync to the cluster
├── config.yaml            # Full experiment configuration
└── requirements.txt
```

---

## Quickstart

### Local run (single experiment)

```bash
pip install -r requirements.txt

# Run one experiment: TTN, ideal, synthetic pre-training, seed 42
python runner.py --config config.yaml \
    --ansatz ttn --noise ideal --source synthetic_gaussian --seed 42
```

### Full pipeline on Hercules

```bash
# 1. Sync to the cluster
bash deploy.sh

# 2. SSH into Hercules and set up the environment (first time only)
conda create -n qcl python=3.10 -y
source activate qcl
pip install -r requirements.txt

# 3. Launch the HUB
python manager.py
# [R] refresh command files
# [F] launch full pipeline (all 60 runs as SLURM array jobs)
# [M] monitor progress
# [T] generate LaTeX tables once complete
```

The SLURM template uses partition `standard`, 4 CPUs, 16 GB RAM, and 12 h time limit.

---

## Configuration

All experiment parameters live in `config.yaml`. The relevant sections are:

- **`ansatze`** — circuit architectures to evaluate
- **`noise_models`** — ideal simulator or IBM Heron r2 calibration
- **`sources`** — initialization strategies (scratch, synthetic Gaussian, MobileNet-V2)
- **`phases`** — SLURM job groups with filter rules
- **`labels`** — LaTeX display names used in generated tables

To adapt this framework for a different paper, create a new `config.yaml` with your own phases and labels, replace `data/loader.py` and `circuits/ansatz.py` with your task-specific code, and keep `manager.py`, `slurm_generic.sh`, and `generate_tables.py` unchanged.

---

## Noise model

The IBM Heron r2 noise model applies three channels after the variational block of each circuit:

| Channel | Parameter | Value (Heron r2) |
|---|---|---|
| Amplitude damping | γ = 1 − exp(−t_g / T1) | ~1.3 × 10⁻⁴ |
| Phase damping | γ = 1 − exp(−t_g · (1/T2 − 1/2T1)) | ~7.4 × 10⁻⁵ |
| Depolarizing | p | 2 × 10⁻⁴ |

These are computed from T1 = 250 µs, T2 = 150 µs, gate time t_g = 32 ns. Noisy simulations use the `default.mixed` density-matrix backend with `diff_method=backprop`.

---

## Results summary

*(Full results available after running on Hercules)*

The key finding is that synthetic Gaussian pre-training reduces the forgetting drop ΔA by over 25% relative to random initialization, without requiring any form of memory buffer. TTN circuits show the lowest ΔA across all conditions. Under IBM Heron r2 noise, the relative ordering across architectures is preserved but absolute accuracy values drop by 5–8 percentage points.

---

## Dependencies

- Python 3.10+
- PennyLane ≥ 0.40.0
- PyTorch ≥ 2.2.0
- scikit-learn ≥ 1.4.0

See `requirements.txt` for the full pinned list.

---

## Citation

If you use this framework, please cite the paper:

```
D. Martín-Pérez, F. Rodríguez-Díaz, D. Gutiérrez-Avilés, A. Troncoso, F. Martínez-Álvarez.
Cross-Domain Pre-Training to Mitigate Catastrophic Forgetting in Sequential Variational Quantum Classifiers.
QUANTICS 2026, Porto.
```
