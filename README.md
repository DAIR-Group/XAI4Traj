# Traj-XAI: Explainable AI for Trajectory Data

A Python package for applying explainable AI (XAI) techniques to trajectory data. This package provides tools for segmenting trajectories, applying perturbations, and generating explanations for black box models.

## Features

- **Trajectory Segmentation**: Multiple methods for dividing trajectories into meaningful segments
  - RDP (Ramer-Douglas-Peucker) segmentation
  - MDL (Minimum Description Length) segmentation
  - Sliding window approach
  - Random segmentation

- **Trajectory Perturbation**: Methods to apply controlled modifications to trajectory segments
  - Gaussian noise perturbation
  - Scaling perturbation
  - Rotation perturbation

- **Model Explanation**: Generate explanations for trajectory classifications from black box models

- **Evaluation**: Tools for evaluating the quality of trajectory explanations

## Installation

```bash
# Clone the repository
git clone https://github.com/buidangphuc/XAI4Traj.git
cd traj-xai

# Install the package
pip install -e .
```

## Quick Start

```python
from pactus import Dataset
from pactus.models import LSTMModel
from traj_xai import (
    rdp_segmentation,
    gaussian_perturbation,
    run_experiments
)

# Load a dataset
dataset = Dataset.uci_movement_libras()
train, test = dataset.split(.8, random_state=0)

# Train a model
model = LSTMModel(random_state=0)
model.train(train, dataset, epochs=10, batch_size=64)

# Run XAI experiments
segment_funcs = [rdp_segmentation]
perturbation_funcs = [gaussian_perturbation]
results = run_experiments(test, segment_funcs, perturbation_funcs, model)
```

## Examples

See the `notebooks` directory for example usage:

- `basic_example.ipynb`: Demonstrates basic usage of the package
- `comparison.ipynb`: Compares different segmentation and perturbation methods

## Requirements

See `requirements.txt` for detailed dependencies.

## Citation

If you use this package in your research, please cite:

```
@software{traj_xai,
  author = {Le Xuan Tung, Bui Dang Phuc},
  title = {Traj-XAI: Explainable AI for Trajectory Data},
  url = {https://github.com/buidangphuc/XAI4Traj.git},
  year = {2025},
}
```
