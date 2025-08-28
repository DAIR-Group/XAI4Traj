import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRUTrajectoryClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=5, num_classes=3):
        super(GRUTrajectoryClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        # If lengths are not provided, assume all sequences are full length
        if lengths is None:
            # Use the full sequence
            batch_size = x.size(0)
            lengths = torch.tensor([x.size(1)] * batch_size, device=x.device)
        
        # Sort sequences by length in descending order
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]
        
        # Pack padded sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, hidden = self.gru(packed)
        
        # Use the last layer's hidden state
        out = self.fc(hidden[-1])
        
        # Unsort the results
        _, unsort_idx = sort_idx.sort(0)
        out = out[unsort_idx]
        
        return out

def build_model(n_classes, input_size=3, hidden_size=64, num_layers=5):
    """
    Helper function to build a GRU trajectory classifier
    
    Args:
        n_classes: Number of classes for classification
        input_size: Size of input features (default 3 for x, y, t)
        hidden_size: Size of hidden layers
        num_layers: Number of recurrent layers
        
    Returns:
        A GRUTrajectoryClassifier model
    """
    return GRUTrajectoryClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=n_classes
    )
