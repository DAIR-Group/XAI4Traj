import logging
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

import pactus.config as cfg
from pactus import Dataset
from pactus.dataset import Data
from pactus.models import Model
from pactus.models.evaluation import Evaluation
from .simple_gru import GRUTrajectoryClassifier, build_model

NAME = "simple_gru_model"
DEFAULT_OPTIMIZER_PARAMS = {'lr': 1e-3}
DEFAULT_PATIENCE = 10

class SimpleGRUModel(Model):
    """Implementation of a simple GRU model for trajectory classification."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 5,
        loss="cross_entropy",
        optimizer_params=None,
        metrics=None,
        max_traj_len: int = -1,
        skip_long_trajs: bool = False,
        mask_value=cfg.MASK_VALUE,
        random_state: Union[int, None] = None,
    ):
        super().__init__(NAME)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model: nn.Module = None
        self.loss_fn = nn.CrossEntropyLoss() if loss == "cross_entropy" else loss
        self.optimizer_params = DEFAULT_OPTIMIZER_PARAMS if optimizer_params is None else optimizer_params
        self.metrics = ["accuracy"] if metrics is None else metrics
        self.max_traj_len = max_traj_len
        self.skip_long_trajs = skip_long_trajs
        self.mask_value = mask_value
        self.encoder: Union[LabelEncoder, None] = None
        self.labels: Union[List[Any], None] = None
        self.original_data: Union[Data, None] = None
        self.random_state: Union[int, None] = random_state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.set_summary(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            loss=self.loss_fn.__class__.__name__,
            optimizer_params=self.optimizer_params,
            metrics=self.metrics,
            max_traj_len=self.max_traj_len,
            skip_long_trajs=self.skip_long_trajs,
        )

    def train(
        self,
        data: Data,
        original_data: Data,
        training=True,  # For compatibility with parent class
        cross_validation: int = 0,
        epochs: int = 20,
        validation_split: float = 0.2,
        batch_size: int = 32,
        callbacks: Union[list, None] = None,
        checkpoint: Union[str, None] = None,
    ):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            logging.warning(
                f"Custom seed provided for {self.name} model. This "
                "sets random seeds for python, numpy and PyTorch."
            )
        
        self.set_summary(
            cross_validation=cross_validation,
            epochs=epochs,
            validation_split=validation_split,
            batch_size=batch_size,
        )
        
        self.encoder = None
        self.labels = data.labels
        self.original_data = original_data
        x_train, y_train, lengths = self._get_input_data(data)
        n_classes = len(data.classes)
        
        # Convert numpy arrays to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(self.device)  # Using indices directly for CrossEntropyLoss
        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(self.device)
        
        dataset = TensorDataset(x_train_tensor, y_train_tensor, lengths_tensor)
        
        model_path = None
        if checkpoint is not None and Path(checkpoint).exists():
            logging.info("Loading model from checkpoint %s", checkpoint)
            model_path = checkpoint

        if cross_validation == 0:
            # Create train/validation split
            val_size = int(len(dataset) * validation_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Build or load model
            if model_path is None:
                model = self._get_model(n_classes)
            else:
                model = self._get_model(n_classes)
                model.load_state_dict(torch.load(model_path))
            
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), **self.optimizer_params)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for inputs, targets, seq_lengths in train_loader:
                    optimizer.zero_grad()
                    
                    try:
                        outputs = model(inputs, seq_lengths)
                        loss = self.loss_fn(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += targets.size(0)
                        train_correct += (predicted == targets).sum().item()
                    except RuntimeError as e:
                        logging.error(f"Error during training: {e}")
                        logging.error(f"Input shape: {inputs.shape}")
                        raise
                
                train_loss /= len(train_loader)
                train_acc = train_correct / train_total
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, targets, seq_lengths in val_loader:
                        outputs = model(inputs, seq_lengths)
                        loss = self.loss_fn(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = val_correct / val_total
                
                logging.info(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss:.4f}, acc: {train_acc:.4f} - Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= DEFAULT_PATIENCE:
                        logging.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Restore best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            self.model = model
            
        else:
            # K-fold cross validation implementation
            # ...existing code for cross validation...
            pass

    def predict(self, data: Data) -> List[Any]:
        self.model.eval()
        x_data, _, lengths = self._get_input_data(data)
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x_tensor, lengths_tensor)
            # Apply softmax to convert logits to probabilities
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
        # Return probabilities as numpy array
        return probabilities.cpu().numpy()

    def _get_model(self, n_classes: int) -> nn.Module:
        """
        Creates the GRU model
        """
        model = build_model(
            n_classes=n_classes,
            input_size=3,  # x, y, t
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        return model

    def _get_input_data(self, data: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process all the data and returns x_data, y_data and lengths
        """
        y_data = self._encode_labels(data)
        x_data, lengths = self._extract_raw_data(data)
        return x_data, y_data, lengths

    def _encode_labels(self, data: Data) -> np.ndarray:
        """Encode the labels as integers for CrossEntropyLoss"""
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit(self.labels)
        encoded_labels = self.encoder.transform(data.labels)
        assert isinstance(encoded_labels, np.ndarray)
        
        # Return integer labels (not one-hot encoded)
        return encoded_labels

    def _extract_raw_data(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts the raw data from the trajectories
        Returns:
            Tuple containing:
            - padded trajectory data [batch, max_len, features]
            - sequence lengths for each trajectory
        """
        assert self.original_data is not None, "Original data must be set"

        trajs = data.trajs
        max_len = max([len(traj) for traj in self.original_data.trajs])
        if self.max_traj_len > 0:
            max_len = min(max_len, self.max_traj_len)
            
        # Extract features [x, y, t]
        raw_data = [np.hstack((traj.r, np.reshape(traj.t, (-1, 1)))) for traj in trajs]
        
        # Filter long trajectories if needed
        if self.skip_long_trajs:
            raw_data = [traj for traj in raw_data if traj.shape[0] <= max_len]
            
        assert len(raw_data) > 0, "No trajectories to train on"
        
        # Store original sequence lengths
        lengths = np.array([min(len(traj), max_len) for traj in raw_data])
        
        # Create padded data
        all_raw_data = np.zeros((len(raw_data), max_len, 3))
        all_raw_data.fill(self.mask_value)
        
        for i, traj in enumerate(raw_data):
            seq_len = min(traj.shape[0], max_len)
            all_raw_data[i, :seq_len] = traj[:seq_len]
            
        return all_raw_data, lengths

    def evaluate(self, data: Data) -> Evaluation:
        assert self.encoder is not None, "Encoder is not set."
        x_data, _, lengths = self._get_input_data(data)
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        lengths_tensor = torch.tensor(lengths, dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor, lengths_tensor)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
        
        evals = self.encoder.inverse_transform(preds)
        return Evaluation.from_data(data, evals, self.summary)
