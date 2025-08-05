# Bio-Inspired Swarm Fault Tolerance
*Scalable and resilient multi-robot simulations*

This repository contains the implementation of bio-inspired fault tolerance mechanisms for robotic swarms, comparing hormone-based and voting-based approaches for autonomous fault detection and quarantine.

## Overview

This project explores how robotic swarms stay resilient under stress. 
We simulate hundreds of robots, introduce failures, and compare different 
strategies for keeping the swarm functional from bio-inspired hormone signaling 
to distributed voting. We implement and compare three approaches:

1. **Hormone-based with quarantine**: Inspired by biological immune responses, using stress hormone propagation
2. **Hormone-based without quarantine**: Same as above, but without quarantine mechanisms
3. **Voting-based quarantine**: A distributed consensus mechanism where robots vote on suspicious neighbors

## Key Features

- **Scale matters**: Run experiments with swarms of 40–200+ robots.
- **Realistic fault modeling**: From minor glitches to severe breakdowns.
- **Stress-tested**: Evaluate performance under packet loss and sensor noise.
- **Full pipeline**: From simulation runs to statistical analysis and figures.
- **Bridging sim & reality**: Includes partial tests in PX4 SITL + Gazebo.


## Installation

### Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- PyYAML
- asyncio

### Setup

```bash

git clone https://github.com/icra2026ano/swarm-fault-tolerance.git 
cd swarm-fault-tolerance 
pip install -r requirements.txt
```

## Project Structure

```
swarm-fault-tolerance/
├── src/
│   ├── core/            # Core source codes
│   │   ├── drone.py     # Base implementation
│   │   ├── simulator.py # Base simulator framework
│   │   └── metrics.py   # Metrics
│   ├── methods/         # Fault detection methods
│   │   ├── hormone.py   # Hormone-based approach
│   │   ├── voting.py    # Voting-based approach
│   │   └── baseline.py  # No quarantine baseline
│   └── utils/           # Some utilities
├── experiments/         # Experiment scripts
├── analysis/            # Analysis and visualization
├── configs/             # Configurations
└── results/             # Output dumping directory
```

## Quick Start

### TL;DR
Clone → Install deps → Run an experiment.
```bash
git clone https://github.com/icra2026ano/swarm-fault-tolerance.git
cd swarm-fault-tolerance
pip install -r requirements.txt
python experiments/run_main_comparison.py
```

### How to Run Comparisons

```bash
# Run default experiments
python experiments/run_main_comparison.py

# If you need custom configuration
python experiments/run_main_comparison.py --config configs/custom_config.yaml

# Robustness tests
python experiments/run_main_comparison.py --robustness
```

### Run Statistical Trials

```bash
# Run multiple trials for statistical validity
python experiments/run_statistical_trials.py --trials 10
```

### Generate Figures

```bash
python analysis/generate_figures.py
```

## Configuration

Experiments are configured via YAML files. Example configuration:

```yaml
# configs/experiment_configs.yaml
swarm_sizes: [40, 75, 100, 150, 200]
fault_rates: [0.10, 0.15, 0.20]
scaling_modes: ['fixed_arena', 'fixed_density']
methods: ['baseline', 'hormone', 'voting']
run_time: 30.0
random_seed: 42

robustness_tests:
  enabled: true
  packet_loss_rates: [0.0, 0.1, 0.2]
  sensor_noise_levels: [0.0, 0.1, 0.2]
```

## Key Results

### Highlights
- Hormone-based quarantine keeps swarms >80% effective, 
  even with 15% faulty robots.
- Voting-based methods work, but collapse under high packet loss.
- Baseline runs show why doing nothing isn’t an option (up to 30% drop in task completion).

### Robustness

- Hormone method maintains >80% of benefits at 20% packet loss
- Voting method fails above 15% packet loss
- Strong positive correlation (r = 0.95) between swarm density and improvement

### Efficiency

- Energy efficiency improved by 15-25% with quarantine
- Cascade prevention rate >85% in dense swarms
- Average detection time: 3-5 seconds for hormone, 5-8 seconds for voting

## Reproducing Paper Results

### Figure 3: Scaling Analysis
```bash
python analysis/generate_scaling_figure.py
```

### Figure 4: Method Comparison
```bash
python analysis/generate_comparison_figure.py
```

### Figure 5: Robustness Analysis
```bash
python experiments/run_robustness_tests.py
python analysis/generate_robustness_figure.py
```

### Table 4: Statistical Summary
```bash
python experiments/run_statistical_trials.py --trials 10
python analysis/generate_statistics_table.py
```


## Troubleshooting

### Common Issues

1. **Memory usage**: For large swarms (>150 robots), ensure sufficient RAM
2. **Slow simulations**: Use `--quiet` flag to suppress output
3. **Reproducibility**: Always set `random_seed` in configuration

### Performance Tips

- Use fixed arena mode for faster simulations
- Reduce `run_time` for quick tests

---
This repo is designed for reproducibility.
If you encounter any issues, please check the [Troubleshooting](#troubleshooting) section
or open a GitHub issue. Tested on Python 3.8+ with Ubuntu 20.04.



- Use parallel processing for multiple trials

