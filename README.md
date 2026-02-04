# Forward-Model Guided Demonstration Learning

A comprehensive codebase for studying imitation learning with human demonstrator biases and label correction methods.

## Overview

This project implements a complete experimental pipeline for comparing different approaches to learning from imperfect human demonstrations:

- **Simple DAgger**: Traditional DAgger using raw human labels
- **MIND MELD**: Neural network-based label correction with learned demonstrator embeddings
- **Cognitive Model**: Explicit cognitive parameter inference for label correction

The environment is a 2D kinematic driving task where a car-like agent must navigate around obstacles to reach a goal.

## Installation

### Requirements

- Python 3.10+
- CPU only (no GPU required)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd FORGE

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install numpy scipy matplotlib pandas torch tqdm pyyaml
```

## Quick Start

### Run a Quick Test Experiment

```bash
# Quick mode (~5 minutes)
python run_experiments.py --config configs/quick.yaml
```

### Run Full Experiment

```bash
# Full experiment (may take several hours)
python run_experiments.py --config configs/base.yaml
```

### Run Tests

```bash
# All tests
python tests/run_tests.py

# Quick test subset
python tests/run_tests.py --quick

# Verbose output
python tests/run_tests.py -v
```

## Project Structure

```
FORGE/
├── configs/                    # Configuration files (YAML)
│   ├── base.yaml              # Base experiment configuration
│   ├── quick.yaml             # Quick test configuration
│   ├── ablation_mindmeld.yaml # MIND MELD ablations
│   ├── ablation_cognitive.yaml # Cognitive model ablations
│   └── ablation_protocol.yaml # Protocol ablations
│
├── envs/                      # Environment implementation
│   ├── __init__.py
│   └── driving_env.py        # 2D kinematic driving environment
│
├── oracle/                    # Oracle controller (ground truth labels)
│   ├── __init__.py
│   └── oracle_controller.py  # A* planner + Pure Pursuit controller
│
├── humans/                    # Synthetic human demonstrators
│   ├── __init__.py
│   └── synthetic_teacher.py  # Cognitive bias simulation
│
├── data/                      # Dataset generation
│   ├── __init__.py
│   └── generate_dataset.py   # Dataset creation with caching
│
├── methods/                   # Learning methods
│   ├── __init__.py
│   ├── policy.py             # MLP policy implementation
│   ├── dagger.py             # DAgger algorithm
│   ├── mindmeld.py           # MIND MELD implementation
│   └── cognitive_model.py    # Cognitive model implementation
│
├── eval/                      # Evaluation
│   ├── __init__.py
│   ├── metrics.py            # Evaluation metrics
│   └── rollout_eval.py       # Rollout-based evaluation
│
├── experiments/               # Experiment utilities
│   ├── __init__.py
│   └── ablations.py          # Ablation study runner
│
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_environment.py
│   ├── test_oracle.py
│   ├── test_teacher.py
│   ├── test_methods.py
│   └── run_tests.py
│
├── results/                   # Auto-created experiment results
│
├── run_experiments.py         # Main entry point
└── README.md
```

## Methods

### Environment

A 2D kinematic "car-like" point with heading:
- State: `s_t = [x, y, ψ, v]` (position, heading, velocity)
- Control: `u_t = ω` (steering rate), constrained to `[-u_max, u_max]`
- Dynamics:
  - `ψ_{t+1} = ψ_t + dt * ω_t`
  - `x_{t+1} = x_t + dt * v * cos(ψ_t)`
  - `y_{t+1} = y_t + dt * v * sin(ψ_t)`

### Oracle Controller

Two-stage optimal controller:
1. **A* Path Planner**: Computes collision-free waypoints on occupancy grid
2. **Pure Pursuit Controller**: Generates steering commands to follow path

### Synthetic Teacher

Simulates human demonstrators with cognitive biases:
- **Temporal offset (τ)**: Anticipatory or delayed responses
- **Gain (g)**: Over/under correction
- **Deadzone (δ)**: Suppression of small actions
- **Motor noise (σ)**: Action variability
- **Saturation**: Personal control limits
- **Dropout**: Attention lapses

### Learning Methods

1. **Simple DAgger** (C1): Standard DAgger with human labels
2. **DAgger + MIND MELD** (C2): Learns per-demonstrator embeddings to correct labels
3. **DAgger + Cognitive Model** (C3): Infers explicit cognitive parameters for correction

## Configuration

All hyperparameters are configured via YAML files. Key parameters:

```yaml
environment:
  dt: 0.1           # Time step
  v: 1.0            # Velocity
  u_max: 1.0        # Max steering rate
  T_max: 200        # Max episode length

population:
  n_demonstrators: 50
  tau_range: [-5.0, 5.0]   # Timing bias range
  g_range: [0.6, 1.4]      # Gain range
  sigma_range: [0.0, 0.15] # Noise range

dagger:
  n_iterations: 5
  m_rollouts_per_iter: 10
  k_init: 4

mindmeld:
  embedding_dim: 8
  context_window: 5
  lstm_hidden: 32
```

## Running Ablations

### MIND MELD Ablations

```bash
python run_experiments.py --config configs/ablation_mindmeld.yaml
```

- A_MM_1: No personalization (single embedding)
- A_MM_2: Context window size
- A_MM_3: State conditioning
- A_MM_4: Embedding dimension
- A_MM_5: MI regularization weight

### Cognitive Model Ablations

```bash
python run_experiments.py --config configs/ablation_cognitive.yaml
```

- A_Cog_1: Remove de-lag (τ fixed to 0)
- A_Cog_2: Remove gain correction (g fixed to 1)
- A_Cog_3: Remove deadzone modeling
- A_Cog_4: Noise estimation
- A_Cog_5: Correction modes

### Protocol Ablations

```bash
python run_experiments.py --config configs/ablation_protocol.yaml
```

- A_P_1: Calibration data size
- A_P_2: Teacher noise level
- A_P_3: Timing bias severity
- A_P_4: Dynamics shift severity
- A_P_5: Data budget per iteration

## Expected Outputs

After running experiments, the results directory contains:

```
results/experiment_YYYYMMDD_HHMMSS/
├── config.yaml              # Experiment configuration
├── results.json             # Raw results per seed
├── aggregated_results.json  # Mean ± std across seeds
├── detailed_results.csv     # Full results table
├── summary_table.csv        # Condensed summary
├── logs/
├── plots/
│   ├── success_rate_comparison.png
│   └── label_correction_mse.png
├── models/
│   ├── dagger_human_seed42.pt
│   ├── dagger_mm_seed42.pt
│   ├── dagger_cog_seed42.pt
│   └── mindmeld_seed42.pt
└── data/
```

## Evaluation Metrics

### Policy Metrics
- Success rate
- Collision rate
- Mean time to goal
- Mean path length
- Integrated cost
- Control smoothness

### Label Metrics
- MSE to oracle (raw and corrected)
- Timing offset (before/after correction)
- Amplitude bias (before/after correction)

## Test Conditions

1. **In-distribution**: Same distribution as training
2. **Shift A**: New obstacle layouts (same statistics)
3. **Shift B**: Increased obstacle density
4. **Shift C**: Changed dynamics (dt, velocity)

## Hypotheses Tested

- **H1**: Corrected labels reduce MSE vs raw
- **H2**: Cognitive model outperforms MIND MELD under distribution shifts
- **H3**: Performance gap increases with smaller calibration sets
- **H4**: Cognitive parameters correlate with MIND MELD embedding axes

## Determinism and Reproducibility

- All random seeds are logged per run
- NumPy and PyTorch seeds are set for reproducibility
- Datasets are cached with configuration hashes

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.20 | Numerical operations |
| scipy | ≥1.7 | Scientific computing |
| matplotlib | ≥3.4 | Plotting |
| pandas | ≥1.3 | Data analysis |
| torch | ≥1.9 | Neural networks |
| tqdm | ≥4.60 | Progress bars |
| pyyaml | ≥5.4 | Configuration |

## Citation

If you use this codebase, please cite:

```bibtex
@software{forward_model_learning,
  title={Forward-Model Guided Demonstration Learning},
  year={2024},
  url={https://github.com/...}
}
```

## License

[License information here]
