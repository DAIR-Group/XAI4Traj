"""
XAI methods for trajectory explanations with logging.
"""

import random
import logging
from typing import List, Generator, Union, Any

import numpy as np
from fastdtw import fastdtw
from pactus import Dataset
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LogisticRegression
from yupi import Trajectory


# -------------------------------------------------------------------
# Logger setup
# -------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)


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
            Z_trajs = [Trajectory(points=np.array(Z_traj)) for Z_traj in self._perturbed_traj_generator()]
            labels = [1] * (len(Z_trajs) - 1) + [0]
            Z_pro = Dataset("custom", Z_trajs, labels)

            preds = self._predict(Z_pro)
            logger.debug(f"Predictions: {preds}, type={type(preds)}")

            pred_labels = self._normalize_predictions(preds)
            Y = self._decode_labels(pred_labels)

            logger.debug(f"Decoded labels (first 10): {Y[:10]}")

            if len(np.unique(Y)) == 1:
                logger.warning("Only one class detected, skipping explanation.")
                return None

            self._fit_surrogate(Y)
            logger.info("Surrogate model fitted successfully.")
            return self.coef_

        except Exception as e:
            logger.error(f"Error in explain: {e}", exc_info=True)
            raise

    def _predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.model, "predict"):
            raise AttributeError("Model must implement predict")
        return self.model.predict(X)

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

    def _decode_labels(self, pred_labels: np.ndarray) -> list:
        if hasattr(self.model, "encoder"):
            return self.model.encoder.inverse_transform(pred_labels)
        if hasattr(self.model, "classes_"):
            return [self.model.classes_[int(label)] for label in pred_labels]
        return pred_labels.tolist() if isinstance(pred_labels, np.ndarray) else list(pred_labels)

    def _fit_surrogate(self, Y: list) -> None:
        clf = LogisticRegression()
        weights = self._calculate_weight()

        if Y and isinstance(Y[0], str):
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
            Z_trajs = [Trajectory(points=np.array(self.X))]
            labels = [0]
            Z_pro = Dataset("custom1", Z_trajs, labels)

            preds = self._predict(Z_pro)
            pred_labels = self._normalize_predictions(preds)
            Y = self._decode_labels(pred_labels)

            if isinstance(Y, np.ndarray):
                return Y.tolist()
            return list(Y)

        except Exception as e:
            logger.error(f"Error in get_Y: {e}", exc_info=True)
            return []

    def get_segment(self) -> List[List[float]]:
        return self.segments
