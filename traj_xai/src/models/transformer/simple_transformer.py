import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Handle attention mask
        attn_mask = None
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # True values are positions to mask
        
        # Apply multi-head attention with residual connection
        attended, _ = self.attention(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout(attended)
        
        # Apply feed-forward network with residual connection
        x = x + self.dropout(self.ff_network(self.norm2(x)))
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(
        self, 
        n_classes, 
        input_dim=3, 
        embed_dim=64, 
        num_heads=4, 
        ff_dim=128, 
        num_layers=2, 
        dropout=0.1,
        mask_value=None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.mask_value = mask_value
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Output head
        self.output_layer = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        # Create mask for padding if mask_value is specified
        mask = None
        if self.mask_value is not None:
            mask = (x != self.mask_value).any(dim=-1)
        
        # Handle different input shapes
        if len(x.shape) == 4:  # [batch, seq, features, channels]
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.reshape(batch_size, seq_len, -1)
        
        # Project input to embedding dimension
        x = self.input_projection(x)
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Apply output projection with softmax
        output = self.output_layer(x)
        return F.softmax(output, dim=-1)

def build_model(
    n_classes,
    input_dim=3,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    dropout=0.1,
    mask_value=None
):
    """
    Helper function to build a simple transformer model
    
    Args:
        n_classes: Number of classes for classification
        input_dim: Input feature dimension
        embed_dim: Hidden dimension of the transformer
        num_heads: Number of attention heads
        ff_dim: Feed-forward network hidden dimension
        num_layers: Number of transformer blocks
        dropout: Dropout rate
        mask_value: Value used for padding (for masking)
        
    Returns:
        A SimpleTransformer model
    """
    return SimpleTransformer(
        n_classes=n_classes,
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        dropout=dropout,
        mask_value=mask_value
    )
