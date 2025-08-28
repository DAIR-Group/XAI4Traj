"""
Experiment execution logic for trajectory XAI.
"""

import os
import numpy as np
import logging

from .evaluation import ap_at_k
from .utils import check_ram_and_log, generate_unique_name, save_result_row
from .xai import TrajectoryManipulator


logger = logging.getLogger(__name__)


def experiment(dataset, segment_func, perturbation_func, blackbox_model):
    """
    Run experiment on a dataset using the specified segmentation and perturbation functions.

    Parameters:
        dataset: The dataset to run experiments on
        segment_func (callable): Function for trajectory segmentation
        perturbation_func (callable): Function for trajectory perturbation
        blackbox_model: The model to explain
    Yields:
        tuple: (trajectory_index, trajectory_name, change_flag, precision_score, status)
    """
    for traj_idx, (traj, label) in enumerate(zip(dataset.trajs, dataset.labels)):
        traj_points = getattr(traj, "r", None)

        if traj_points is None or len(traj_points) == 0:
            logger.error(f"Trajectory {traj_idx} is empty or None. Skipping...")
            yield traj_idx, None, None, None, "error_empty"
            continue

        traj_name = generate_unique_name(traj_points)

        try:
            trajectory_experiment = TrajectoryManipulator(
                traj_points, segment_func, perturbation_func, blackbox_model
            )
        except Exception as e:
            logger.error(f"Init error at trajectory {traj_idx}: {e}", exc_info=True)
            yield traj_idx, traj_name, None, None, "error_init"
            continue

        try:
            coef = trajectory_experiment.explain()
            if coef is None:
                logger.info(f"Trajectory {traj_idx}: classification unchanged")
                yield traj_idx, traj_name, 0, 0.0, "ok"
                continue
        except Exception as e:
            logger.error(f"Explain error at trajectory {traj_idx}: {e}", exc_info=True)
            yield traj_idx, traj_name, None, None, "error_explain"
            continue

        try:
            trajectory_experiment.get_segment()
        except Exception as e:
            logger.error(f"Segment error at trajectory {traj_idx}: {e}", exc_info=True)
            yield traj_idx, traj_name, None, None, "error_segment"
            continue

        try:
            relevant_class = trajectory_experiment.get_Y()
            if not relevant_class:
                logger.error(f"Trajectory {traj_idx}: prediction failed")
                yield traj_idx, traj_name, None, None, "error_predict"
                continue
        except Exception as e:
            logger.error(f"Predict error at trajectory {traj_idx}: {e}", exc_info=True)
            yield traj_idx, traj_name, None, None, "error_predict"
            continue

        try:
            y_true = trajectory_experiment.get_Y_eval_sorted()
            if not y_true:
                logger.error(f"Trajectory {traj_idx}: perturbed outputs missing")
                yield traj_idx, traj_name, None, None, "error_eval"
                continue
        except Exception as e:
            logger.error(f"Eval error at trajectory {traj_idx}: {e}", exc_info=True)
            yield traj_idx, traj_name, None, None, "error_eval"
            continue

        # Compute change
        try:
            change = 0
            for item in y_true:
                is_in_relevant = any(
                    (
                        isinstance(item, np.ndarray)
                        and isinstance(cls, np.ndarray)
                        and np.array_equal(item, cls)
                    )
                    or (
                        hasattr(item, "shape")
                        and hasattr(cls, "shape")
                        and np.all(item == cls)
                    )
                    or (item == cls)
                    for cls in relevant_class
                )
                if not is_in_relevant:
                    change = 1
                    break
        except Exception as e:
            logger.error(f"Change error at trajectory {traj_idx}: {e}", exc_info=True)
            yield traj_idx, traj_name, None, None, "error_change"
            continue

        try:
            precision_score = (
                ap_at_k(y_true, relevant_class, len(y_true)) if change else 0.0
            )
        except Exception as e:
            logger.error(
                f"Precision error at trajectory {traj_idx}: {e}", exc_info=True
            )
            yield traj_idx, traj_name, change, None, "error_precision"
            continue

        yield traj_idx, traj_name, change, precision_score, "ok"


def run_experiments(dataset, segment_funcs, perturbation_funcs, model, log_dir="logs"):
    """
    Run multiple experiments with different segmentation and perturbation functions.

    Parameters:
        dataset: The dataset to run experiments on
        segment_funcs (list): List of segmentation functions
        perturbation_funcs (list): List of perturbation functions
        model: The model to explain
        log_dir (str): Directory for log files
    """
    os.makedirs(log_dir, exist_ok=True)

    # Loop through segmentation and perturbation functions
    for segment_func in segment_funcs:
        for perturbation_func in perturbation_funcs:
            # Generate file path for saving results
            file_path = os.path.join(
                log_dir,
                f"{segment_func.__name__}_{perturbation_func.__name__}_results.csv",
            )

            logger.info(
                f"Running experiment with {segment_func.__name__} and {perturbation_func.__name__}"
            )

            # Loop through the experiment results and save row by row
            for result in experiment(dataset, segment_func, perturbation_func, model):
                traj_idx, traj_name, change, precision_score, status = result

                # Save each row to the CSV
                save_result_row(
                    [traj_idx, traj_name, change, precision_score, status], file_path
                )

                # Check RAM usage periodically
                if traj_idx % 10 == 0:
                    if check_ram_and_log(ram_limit=80, log_dir=log_dir):
                        logger.warning("RAM usage too high. Pausing experiment...")
                        break

            logger.info(f"Results saved to {file_path}")
