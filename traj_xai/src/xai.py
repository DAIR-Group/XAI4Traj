"""
XAI methods for trajectory explanations with logging.
"""

import random
import logging
import torch
from typing import List, Generator, Union, Any

import numpy as np
from fastdtw import fastdtw
from pactus import Dataset
from pactus.dataset import Data
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from yupi import Trajectory


# -------------------------------------------------------------------
# Logger setup
# -------------------------------------------------------------------

# Ghi log ra file thay vì terminal
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    fh = logging.FileHandler('traj_xai_log.txt', mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class TrajectoryManipulator:
    """
    Manipulates trajectories to explain model predictions using XAI techniques.
    """

    def __init__(
        self,
        X: List[List[float]],
        segmentation_model: Any,
        perturbation_model: Any,
        model: Any,
    ) -> None:
        self.X: List[List[float]] = list(X)
        self.segmentation_model = segmentation_model
        self.perturbation_model = perturbation_model
        self.model = model

        self.segments: List[List[float]] = self._segmentation(self.X)
        self.x_len: int = len(self.segments)
        self.number_of_permutations: int = min(2**10, 2**self.x_len)

        self.perturb_vectors: List[List[int]] = self.create_perturbation_points_by_shuffle(
            self.x_len, self.number_of_permutations
        )

        self.clean_segments: List[List[float]] = self.segments
        self.noisy_segments: List[List[float]] = [self._perturbation(seg) for seg in self.segments]

        self.Z_eval: List[List[float]] = self._createZForEval()

    def _segmentation(self, points_list: List[List[float]]) -> List[List[float]]:
        return self.segmentation_model(points_list)

    def _perturbation(self, segment: List[List[float]]) -> List[List[float]]:
        return self.perturbation_model(segment)

    @staticmethod
    def create_perturbation_points_by_shuffle(vector_length: int, samples: int) -> List[List[int]]:
        return [
            [random.randint(0, 1) for _ in range(vector_length)]
            for _ in range(samples)
        ]

    def _convert_perturb_vector_to_traj(self, vector: List[int]) -> List[List[float]]:
        return sum(
            [
                self.noisy_segments[i] if bit == 1 else self.clean_segments[i]
                for i, bit in enumerate(vector)
            ],
            [],
        )

    def _perturbed_traj_generator(self) -> Generator[List[List[float]], None, None]:
        for vector in self.perturb_vectors:
            yield self._convert_perturb_vector_to_traj(vector)

    def _createZForEval(self) -> List[List[float]]:
        identity_matrix = [
            [1 if i == j else 0 for j in range(self.x_len)]
            for i in range(self.x_len)
        ]
        return [
            sum(
                [
                    self.noisy_segments[i] if bit == 1 else self.clean_segments[i]
                    for i, bit in enumerate(row)
                ],
                [],
            )
            for row in identity_matrix
        ]

    def calc_dtw(self, raw: List[List[float]]) -> List[float]:
        return [
            fastdtw(raw, pert, dist=euclidean)[0]
            for pert in self._perturbed_traj_generator()
        ]

    def _calculate_weight(self) -> List[float]:
        distances = self.calc_dtw(self.X)
        mean_dist = sum(distances) / len(distances)
        std_distances = (
            sum((x - mean_dist) ** 2 for x in distances) / len(distances)
        ) ** 0.5
        weights = [
            1
            if std_distances == 0
            else (np.e ** (-abs((d - mean_dist) / (std_distances + 1e-10))))
            for d in distances
        ]
        return weights

    def explain(self) -> Union[np.ndarray, None]:
        try:
            logger.debug("Starting explanation process...")
            Z_trajs = [Trajectory(points=np.array(Z_traj)) for Z_traj in self._perturbed_traj_generator()]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom", Z_trajs, labels)
            preds = self._predict(Z_pro)

            pred_labels = self._normalize_predictions(preds)
            Y = self._decode_labels(pred_labels)

            if len(np.unique(Y)) == 1:
                logger.warning("Only one class detected, skipping explanation.")
                return None

            self._fit_surrogate(Y)
            logger.info("Finished explanation process.")
            return self.coef_

        except Exception as e:
            logger.error(f"Error in explain: {e}", exc_info=True)
            raise

    def _predict(self, X: np.ndarray) -> np.ndarray:
        try:
            result = self.model.predict(X)
            return result
        except Exception as e:
            logger.error(f"Error in _predict: {e}", exc_info=True)
            raise

    def _normalize_predictions(self, preds: Union[np.ndarray, tuple]) -> np.ndarray:
        logger.debug(f"Normalizing predictions: type={type(preds)}, shape={getattr(preds,'shape',None)}")

        if isinstance(preds, tuple):
            preds = preds[0]

        if isinstance(preds, np.ndarray):
            if preds.ndim == 1:
                return preds
            if preds.ndim == 2 and preds.shape[1] > 1:
                return np.argmax(preds, axis=1)
            if preds.ndim == 2 and preds.shape[1] == 1:
                return preds.ravel()

        raise ValueError(f"Unsupported prediction format: {type(preds)}, shape={getattr(preds,'shape',None)}")
        
    # def _prepare_data_for_trajformer(self, dataset):
    #     """
    #     Prepare data in the format expected by TrajFormer model.
    #     Returns:
    #         x_trajs: tensor of trajectory features
    #         masks: attention masks
    #         distances: distance matrices for CPE
    #     """
    #     logger.debug(f"Preparing data for TrajFormer: dataset has {len(dataset.trajs)} trajectories")
        
    #     try:
    #         import torch
    #         import numpy as np
            
    #         # Extract parameters from the model
    #         if hasattr(self.model, 'max_points'):
    #             max_points = self.model.max_points
    #         elif hasattr(self.model, 'model') and hasattr(self.model.model, 'max_points'):
    #             max_points = self.model.model.max_points
    #         else:
    #             max_points = 100  # Default value
                
    #         if hasattr(self.model, 'c_in'):
    #             c_in = self.model.c_in
    #         elif hasattr(self.model, 'model') and hasattr(self.model.model, 'c_in'):
    #             c_in = self.model.model.c_in
    #         else:
    #             c_in = 6  # Default value
                
    #         all_features = []
    #         all_masks = []
    #         all_distances = []
            
    #         for traj in dataset.trajs:
    #             # Extract coordinates and time
    #             coords = np.array(traj.r)  # coordinates
    #             times = np.array(traj.t) if hasattr(traj, 't') and traj.t is not None else np.arange(len(coords))
                
    #             # Calculate features
    #             features = self._extract_features_for_trajformer(coords, times, c_in)
                
    #             # Pad or truncate to max_points
    #             if len(features) > max_points:
    #                 features = features[:max_points]
                    
    #             # Create mask (False for actual data, True for padding)
    #             mask = torch.zeros(max_points, dtype=torch.bool)
    #             if len(features) < max_points:
    #                 # Pad features
    #                 padding = np.zeros((max_points - len(features), c_in))
    #                 features = np.vstack([features, padding])
    #                 # Set mask for padded values
    #                 mask[len(features):] = True
                    
    #             # Calculate distance matrix for CPE
    #             dist_matrix = self._calculate_distances_for_trajformer(coords, max_points)
                
    #             all_features.append(features)
    #             all_masks.append(mask)
    #             all_distances.append(dist_matrix)
                
    #         # Convert to tensors
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         return (
    #             torch.tensor(np.array(all_features), dtype=torch.float32).to(device),
    #             torch.stack(all_masks).to(device),
    #             torch.tensor(np.array(all_distances), dtype=torch.float32).to(device),
    #         )
            
    #     except Exception as e:
    #         logger.error(f"Error preparing data for TrajFormer: {e}", exc_info=True)
    #         raise
        
    def _extract_features_for_trajformer(self, coords, times, c_in=6):
        """Extract features from trajectory coordinates and times for TrajFormer"""
        n_points = len(coords)
        features = np.zeros((n_points, c_in))
        
        # Set lat, lng
        features[:, 0] = coords[:, 0]  # latitude
        features[:, 1] = coords[:, 1]  # longitude
        
        # Calculate time differences
        if n_points > 1:
            # Use provided times if available
            dt = np.diff(times, prepend=times[0])
            features[:, 2] = dt  # delta time
            
            # Calculate distances between consecutive points
            dx = np.diff(coords[:, 0], prepend=coords[0, 0])
            dy = np.diff(coords[:, 1], prepend=coords[0, 1])
            distances = np.sqrt(dx**2 + dy**2)
            features[:, 3] = distances  # delta distance
            
            # Calculate speeds
            with np.errstate(divide="ignore", invalid="ignore"):
                speeds = np.zeros_like(dt)
                valid_dt = dt > 0
                speeds[valid_dt] = distances[valid_dt] / dt[valid_dt]
            features[:, 4] = speeds  # speed
            
            # Calculate accelerations
            accels = np.diff(speeds, prepend=speeds[0])
            features[:, 5] = accels  # acceleration
            
        return features
        
    def _calculate_distances_for_trajformer(self, coords, max_points):
        """Calculate distance matrix for CPE module of TrajFormer"""
        kernel_size = 9  # Default kernel size used in TrajFormer CPE
        n_points = min(len(coords), max_points)
        distances = np.zeros((max_points, kernel_size, 2))
        
        # For each point, calculate distances to kernel_size neighbors
        half_k = kernel_size // 2
        for i in range(n_points):
            for j in range(kernel_size):
                idx = i - half_k + j
                if 0 <= idx < n_points:
                    # Calculate distance components
                    if i != idx:  # Avoid self-distance calculation issues
                        dx = coords[i, 0] - coords[idx, 0]
                        dy = coords[i, 1] - coords[idx, 1]
                        distances[i, j, 0] = dx
                        distances[i, j, 1] = dy
                        
        return distances

    def _decode_labels(self, pred_labels: np.ndarray) -> list:
        if hasattr(self.model, "encoder"):
            return self.model.encoder.inverse_transform(pred_labels)
        if hasattr(self.model, "classes_"):
            return [self.model.classes_[int(label)] for label in pred_labels]
        return pred_labels.tolist() if isinstance(pred_labels, np.ndarray) else list(pred_labels)

    def _fit_surrogate(self, Y: list) -> None:
        clf = LogisticRegression()
        weights = self._calculate_weight()

        if len(Y) > 0 and isinstance(Y[0], str):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            Y_encoded = le.fit_transform(Y)
            clf.fit(self.perturb_vectors, Y_encoded, sample_weight=weights)
        else:
            clf.fit(self.perturb_vectors, Y, sample_weight=weights)

        self.coef_ = clf.coef_
        self.classes_ = clf.classes_

    def get_Y_eval_sorted(self) -> Union[List[Any], np.ndarray]:
        try:
            logger.debug("Starting get_Y_eval_sorted process...")
            Z_trajs = [Trajectory(points=np.array(Z_traj)) for Z_traj in self.Z_eval]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)

            preds = self._predict(Z_pro)
                
            Y = self._normalize_predictions(preds)

            if Y is None or len(Y) == 0:
                raise ValueError("No predictions returned in get_Y_eval_sorted")

            Y_without_pertub = self.get_Y()
            class_coef = self._get_class_coef(Y_without_pertub)

            sorted_indices = np.argsort(abs(class_coef))[::-1]
            result_sorted: List[Any] = [Y[i] for i in sorted_indices if i < len(Y)]
            logger.debug("Finished get_Y_eval_sorted process.")
            return result_sorted

        except Exception as e:
            logger.error(f"Error in get_Y_eval_sorted: {e}", exc_info=True)
            return np.full(len(self.Z_eval), np.nan)

    def _get_class_coef(self, Y_without_pertub: List[Any]) -> np.ndarray:
        if not hasattr(self, "classes_") or not hasattr(self, "coef_"):
            return np.zeros(len(self.Z_eval))

        if Y_without_pertub:
            target_class = Y_without_pertub[0]
            try:
                if isinstance(target_class, str) and all(isinstance(c, (int, np.integer)) for c in self.classes_):
                    target_class = int(target_class)
                class_index = np.where(self.classes_ == target_class)[0][0]
                return self.coef_[class_index]
            except (ValueError, IndexError, TypeError):
                return self.coef_[0]

        return self.coef_[0]

    def get_Y(self) -> List[Any]:
        try:
            logger.debug("Starting get_Y process...")
            Z_trajs = [Trajectory(points=np.array(self.X))]
            labels = [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)
            # logger.debug(f"get_Y: Z_pro type={type(Z_pro)}, Z_trajs type={type(Z_trajs)}, len(Z_trajs)={len(Z_trajs)}")

            preds = self._predict(Z_pro)
                
            pred_labels = self._normalize_predictions(preds)
            Y = self._decode_labels(pred_labels)
            if isinstance(Y, np.ndarray):
                return Y.tolist()
            logger.debug(f"Finished get_Y process, Y length: {len(Y)}")
            return list(Y)
        except Exception as e:
            logger.error(f"Error in get_Y: {e}", exc_info=True)
            return []

    def get_segment(self) -> List[List[float]]:
        return self.segments
