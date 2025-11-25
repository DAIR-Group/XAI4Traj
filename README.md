# XAI4Traj: Explainable AI for Trajectory Classification

A Python package for applying explainable AI (XAI) techniques to trajectory classification. This package implements a segment-based perturbation approach for explaining trajectory classifications from black box models.

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
git clone https://github.com/DAIR-Group/XAI4Traj.git
cd traj_xai

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

```bibtex
@article{tung2025towards,
  title={Towards explainable trajectory classification: A segment-based perturbation approach},
  author={Tung, Le Xuan and Phuc, Bui Dang and Le Duy, Vo Nguyen},
  journal={Neurocomputing},
  volume={658},
  pages={131691},
  year={2025},
  publisher={Elsevier}
}
```
