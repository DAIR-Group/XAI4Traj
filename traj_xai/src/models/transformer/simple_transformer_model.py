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
from .simple_transformer import build_model

NAME = "simple_transformer_model"
DEFAULT_OPTIMIZER_PARAMS = {'lr': 1e-3}
DEFAULT_PATIENCE = 10


class SimpleTransformerModel(Model):
    """Implementation of a simplified Transformer model for trajectory classification."""

    def __init__(
        self,
        embed_dim: int = 16,
        num_heads: int = 1,
        ff_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.05,
        loss="cross_entropy",
        optimizer_params=None,
        metrics=None,
        max_traj_len: int = -1,
        skip_long_trajs: bool = False,
        mask_value=cfg.MASK_VALUE,
        random_state: Union[int, None] = None,
    ):
        super().__init__(NAME)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.dropout = dropout
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
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
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
        x_train, y_train = self._get_input_data(data)
        n_classes = len(data.classes)
        
        # Convert numpy arrays to PyTorch tensors
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(x_train_tensor, y_train_tensor)
        
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
                
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    
                    try:
                        outputs = model(inputs)
                        loss = self.loss_fn(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        _, target_indices = torch.max(targets, 1)
                        train_total += targets.size(0)
                        train_correct += (predicted == target_indices).sum().item()
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
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        loss = self.loss_fn(outputs, targets)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        _, target_indices = torch.max(targets, 1)
                        val_total += targets.size(0)
                        val_correct += (predicted == target_indices).sum().item()
                
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
            # K-fold cross-validation
            assert cross_validation > 1, "cross_validation must be greater than 1"
            kfold = KFold(n_splits=cross_validation, shuffle=True, random_state=self.random_state)
            
            best_acc = -1
            best_model = None
            fold_no = 1
            
            # Convert tensors back to numpy for KFold
            x_train_np = x_train_tensor.cpu().numpy()
            y_train_np = y_train_tensor.cpu().numpy()
            
            for train_idxs, test_idxs in kfold.split(x_train_np):
                # Create datasets for this fold
                x_train_fold = torch.tensor(x_train_np[train_idxs], dtype=torch.float32).to(self.device)
                y_train_fold = torch.tensor(y_train_np[train_idxs], dtype=torch.float32).to(self.device)
                x_test_fold = torch.tensor(x_train_np[test_idxs], dtype=torch.float32).to(self.device)
                y_test_fold = torch.tensor(y_train_np[test_idxs], dtype=torch.float32).to(self.device)
                
                train_dataset = TensorDataset(x_train_fold, y_train_fold)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                
                # Create new model for this fold
                model = self._get_model(n_classes)
                model = model.to(self.device)
                optimizer = optim.Adam(model.parameters(), **self.optimizer_params)
                
                # Training loop for this fold
                for epoch in range(epochs):
                    model.train()
                    train_loss = 0.0
                    
                    for inputs, targets in train_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = self.loss_fn(outputs, targets)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                
                # Evaluate on the test fold
                model.eval()
                test_loss = 0.0
                test_correct = 0
                test_total = len(test_idxs)
                
                with torch.no_grad():
                    outputs = model(x_test_fold)
                    loss = self.loss_fn(outputs, y_test_fold)
                    test_loss = loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    _, target_indices = torch.max(y_test_fold, 1)
                    test_correct = (predicted == target_indices).sum().item()
                
                acc = test_correct / test_total
                
                logging.info(f"Fold {fold_no} - Loss: {test_loss:.4f}, Accuracy: {acc:.4f}")
                
                # Save the best model
                if acc > best_acc:
                    best_acc = acc
                    best_model = model
                
                fold_no += 1
            
            self.model = best_model

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
        Creates the simple transformer model
        """
        model = build_model(
            n_classes=n_classes,
            input_dim=3,  # x, y, t
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            mask_value=self.mask_value
        )
        return model

    def _get_input_data(self, data: Data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all the data and returns a x_data, y_data readable
        by the transformer
        """
        y_data = self._encode_labels(data)
        x_data = self._extract_raw_data(data)
        return x_data, y_data

    def _encode_labels(self, data: Data) -> np.ndarray:
        """Encode the labels"""
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit(self.labels)
        encoded_labels = self.encoder.transform(data.labels)
        assert isinstance(encoded_labels, np.ndarray)

        classes = np.zeros((len(encoded_labels), len(self.encoder.classes_)))
        for i, label in enumerate(encoded_labels):
            classes[i][label] = 1
        return classes

    def _extract_raw_data(self, data: Data) -> np.ndarray:
        """Extracts the raw data from the yupi trajectories"""
        assert self.original_data is not None, "Original data must be set"

        trajs = data.trajs
        max_len = np.max([len(traj) for traj in self.original_data.trajs])
        if self.max_traj_len > 0:
            max_len = min(max_len, self.max_traj_len)
            
        raw_data = [np.hstack((traj.r, np.reshape(traj.t, (-1, 1)))) for traj in trajs]
        if self.skip_long_trajs:
            raw_data = [traj for traj in raw_data if traj.shape[0] <= max_len]
        assert len(raw_data) > 0, "No trajectories to train on"
        
        all_raw_data = np.zeros((len(raw_data), max_len, 3))
        for i, traj in enumerate(raw_data):
            traj = traj[:max_len]
            all_raw_data[i, :, :] = self.mask_value
            all_raw_data[i, : traj.shape[0]] = traj
        return all_raw_data

    def evaluate(self, data: Data) -> Evaluation:
        assert self.encoder is not None, "Encoder is not set."
        x_data, _ = self._get_input_data(data)
        x_tensor = torch.tensor(x_data, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            preds = self.model(x_tensor)
            preds = preds.cpu().numpy()
        
        preds = [pred.argmax() for pred in preds]
        evals = self.encoder.inverse_transform(preds)
        return Evaluation.from_data(data, evals, self.summary)
