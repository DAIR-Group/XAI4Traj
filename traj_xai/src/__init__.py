"""
traj-xai: A package for explainable AI on trajectory data.
"""

from .evaluation import ap_at_k
from .experiment import experiment, run_experiments
from .perturbation import (
    gaussian_perturbation,
    rotation_perturbation,
    scaling_perturbation,
)
from .segmentation import (
    mdl_segmentation,
    random_segmentation,
    rdp_segmentation,
    sliding_window_segmentation,
)
from .utils import check_ram_and_log, generate_unique_name, save_result_row
from .xai import TrajectoryManipulator
from pactus.models import (
    RandomForestModel,
    KNeighborsModel,
)
from .models import (
    SimpleGRUModel,
    SimpleTransformerModel,
    TrajFormerModel,
)
__all__ = [
    "rdp_segmentation",
    "random_segmentation",
    "sliding_window_segmentation",
    "mdl_segmentation",
    "gaussian_perturbation",
    "scaling_perturbation",
    "rotation_perturbation",
    "ap_at_k",
    "TrajectoryManipulator",
    "experiment",
    "run_experiments",
    "save_result_row",
    "check_ram_and_log",
    "generate_unique_name",
    "RandomForestModel",
    "KNeighborsModel",
    "SimpleGRUModel",
    "SimpleTransformerModel",
    "TrajFormerModel",
]
