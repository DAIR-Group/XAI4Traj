import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pactus.dataset import Data
from pactus.models import Model
from pactus.models.evaluation import Evaluation
from sklearn.preprocessing import LabelEncoder
from .trajformer import TrajFormer

NAME = "trajformer_model"


class TrajFormerWrapper(nn.Module):
    """Wrapper for TrajFormer model to handle device compatibility."""
    
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        
    def forward(self, x_trajs, masks, distances):
        # Move inputs to the correct device
        x_trajs = x_trajs.to(self.device)
        masks = masks.to(self.device)
        distances = distances.to(self.device)
        
        # Forward pass through the base model
        return self.model(x_trajs, masks, distances)
    
    def train(self, mode=True):
        self.model.train(mode)
        return self
        
    def eval(self):
        self.model.eval()
        return self


class TrajFormerModel(Model):
    """Implementation of TrajFormer model for trajectory classification."""

    def __init__(
        self,
        c_in=6,
        c_out=4,
        trans_layers=3,
        n_heads=4,
        token_dim=64,
        kv_pool=1,
        mlp_dim=256,
        max_points=100,
        cpe_layers=1,
        metrics=None,
        random_state: Union[int, None] = None,
        device="mps" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(NAME)
        self.c_in = c_in
        self.c_out = c_out
        self.trans_layers = trans_layers
        self.n_heads = n_heads
        self.token_dim = token_dim
        self.kv_pool = kv_pool
        self.mlp_dim = mlp_dim
        self.max_points = max_points
        self.cpe_layers = cpe_layers
        self.metrics = ["accuracy"] if metrics is None else metrics
        self.random_state = random_state
        self.device = device
        self.encoder = None
        self.labels = None
        self.model = None

        # Set summary for evaluation reporting
        self.set_summary(
            c_in=self.c_in,
            c_out=self.c_out,
            trans_layers=self.trans_layers,
            n_heads=self.n_heads,
            token_dim=self.token_dim,
            kv_pool=self.kv_pool,
            mlp_dim=self.mlp_dim,
            max_points=self.max_points,
            cpe_layers=self.cpe_layers,
            metrics=self.metrics,
        )

    def train(self, data: Data, original_data: Data, training=True, **kwargs):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            logging.warning(
                f"Custom seed provided for {self.name} model. This "
                "sets random seeds for python, numpy, and PyTorch."
            )

        # Initialize the encoder and store labels
        self.encoder = LabelEncoder()
        self.labels = data.labels
        y_encoded = self.encoder.fit_transform(self.labels)

        # Create and initialize the base TrajFormer model
        base_model = TrajFormer(
            name=self.name,
            c_in=self.c_in,
            c_out=self.c_out,
            trans_layers=self.trans_layers,
            n_heads=self.n_heads,
            token_dim=self.token_dim,
            kv_pool=self.kv_pool,
            mlp_dim=self.mlp_dim,
            max_points=self.max_points,
            cpe_layers=self.cpe_layers,
            device=self.device,  # Pass the device parameter
        ).to(self.device)

        # Wrap the base model with our wrapper to handle CUDA/CPU compatibility
        self.model = TrajFormerWrapper(base_model, self.device)

        # Convert data to the format expected by TrajFormer
        x_trajs, masks, distances = self._prepare_data(data)

        # Training loop implementation
        self._train_model(x_trajs, masks, distances, y_encoded, **kwargs)

        logging.info(f"Trained TrajFormer model with {len(self.labels)} samples")

    def predict(self, data: Data) -> np.ndarray:
        """
        Predict class probabilities for each trajectory

        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        assert self.model is not None, "Model has not been trained yet"
        assert self.encoder is not None, "Encoder is not initialized"

        # Prepare data
        x_trajs, masks, distances = self._prepare_data(data)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_trajs, masks, distances)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()

        return probabilities

    def evaluate(self, data: Data) -> Evaluation:
        """Evaluate the model on test data"""
        assert self.encoder is not None, "Encoder is not set."

        # Get predicted probabilities
        probabilities = self.predict(data)

        # Get class with highest probability for each sample
        pred_indices = np.argmax(probabilities, axis=1)

        # Convert back to original class labels
        predictions = self.encoder.inverse_transform(pred_indices)

        return Evaluation.from_data(data, predictions, self.summary)

    def _train_model(self, x_trajs, masks, distances, y_encoded, **kwargs):
        """Train the model with the prepared data"""
        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=kwargs.get("learning_rate", 0.001)
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        n_epochs = kwargs.get("n_epochs", 10)
        batch_size = kwargs.get("batch_size", 32)

        # Convert to PyTorch tensors and move to the right device
        y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self.device)

        for epoch in range(n_epochs):
            # Mini-batch training
            total_loss = 0
            for i in range(0, len(x_trajs), batch_size):
                batch_x = x_trajs[i : i + batch_size]  # Move to device inside wrapper
                batch_masks = masks[i : i + batch_size]
                batch_distances = distances[i : i + batch_size]
                batch_y = y_tensor[i : i + batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_x, batch_masks, batch_distances)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")

    def _prepare_data(self, data: Data):
        """
        Convert data from pactus format to TrajFormer input format

        Returns:
            x_trajs: tensor of trajectory features
            masks: attention masks
            distances: distance matrices for CPE
        """
        # Implement data conversion based on actual data structure
        # This is a simplified implementation that should be customized

        all_features = []
        all_masks = []
        all_distances = []

        for traj in data.trajs:
            # Extract coordinates, time, etc.
            coords = np.array(traj.r)  # coordinates
            times = np.array(traj.t)  # timestamps

            # Calculate features (speeds, accelerations, etc.)
            features = self._extract_features(coords, times)

            # Pad or truncate to max_points
            if len(features) > self.max_points:
                features = features[: self.max_points]

            # Create mask (False for actual data, True for padding)
            mask = torch.zeros(self.max_points, dtype=torch.bool)
            if len(features) < self.max_points:
                # Pad features
                padding = np.zeros((self.max_points - len(features), self.c_in))
                features = np.vstack([features, padding])
                # Set mask for padded values
                mask[len(features) :] = True

            # Calculate distance matrix for CPE
            dist_matrix = self._calculate_distances(coords)

            all_features.append(features)
            all_masks.append(mask)
            all_distances.append(dist_matrix)

        # Convert to tensors
        return (
            torch.tensor(np.array(all_features), dtype=torch.float32),
            torch.stack(all_masks),
            torch.tensor(np.array(all_distances), dtype=torch.float32),
        )

    def _extract_features(self, coords, times):
        """Extract features from trajectory coordinates and times"""
        n_points = len(coords)
        features = np.zeros((n_points, self.c_in))

        # Set lat, lng
        features[:, 0] = coords[:, 0]  # latitude
        features[:, 1] = coords[:, 1]  # longitude

        # Calculate time differences
        if n_points > 1 and len(times) == n_points:
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
        else:
            # If no time data or only one point, use synthetic time (equal intervals)
            synthetic_times = np.arange(n_points)
            
            # Recalculate all features with synthetic time
            dt = np.ones(n_points)  # constant time steps
            features[:, 2] = dt  # delta time
            
            # Calculate distances
            if n_points > 1:
                dx = np.diff(coords[:, 0], prepend=coords[0, 0])
                dy = np.diff(coords[:, 1], prepend=coords[0, 1])
                distances = np.sqrt(dx**2 + dy**2)
                features[:, 3] = distances  # delta distance
                
                # Use distances as speed (since dt=1)
                features[:, 4] = distances  # speed
                
                # Calculate accelerations
                accels = np.diff(distances, prepend=distances[0])
                features[:, 5] = accels  # acceleration

        return features

    def _calculate_distances(self, coords):
        """Calculate distance matrix for CPE module"""
        n_points = min(len(coords), self.max_points)
        kernel_size = 9
        distances = np.zeros((self.max_points, kernel_size, 2))

        # For each point, calculate distances to kernel_size neighbors
        half_k = kernel_size // 2
        for i in range(n_points):
            for j in range(kernel_size):
                idx = i - half_k + j
                if 0 <= idx < n_points:
                    # Calculate distance components
                    distances[i, j, 0] = coords[idx, 0] - coords[i, 0]  # delta lat
                    distances[i, j, 1] = coords[idx, 1] - coords[i, 1]  # delta lng

        return distances
