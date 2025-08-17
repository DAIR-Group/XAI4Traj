import logging
from abc import abstractmethod
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from einops import repeat
from fvcore.nn import FlopCountAnalysis
from pactus.dataset import Data
from pactus.models import Model
from pactus.models.evaluation import Evaluation
from timm.models.layers import trunc_normal_

NAME = "trajformer"


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def param_num(self, str):
        return sum([param.nelement() for param in self.parameters()])

    def flops(self, inputs):
        flops = FlopCountAnalysis(self, inputs)
        return flops.total()

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


class PreNorm(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm_Mask(nn.Module):

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, mask, dt, dd):
        return self.fn(self.norm(x), mask, dt, dd)


class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PoolAttention(nn.Module):

    def __init__(
        self,
        dim,
        heads=4,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.0,
        kv_pool=1,
        stage=None,
    ):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.kv_pool = kv_pool
        self.num_heads = heads
        self.stage = stage
        head_dim = dim // heads
        self.scale = qk_scale or head_dim**-0.5

        if kv_pool > 1:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.pooling = nn.MaxPool1d(kv_pool, kv_pool, 0)
        elif kv_pool == 1:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        else:
            raise ValueError("kv pool rate should be greater than 0!")

        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask, dt, dd):
        B, N, C = x.shape

        if self.kv_pool > 1:
            q = (
                self.q(x)
                .reshape(B, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            pooled_kv = self.pooling(x[:, 1:].permute(0, 2, 1))
            pre_kv = torch.cat([pooled_kv, x[:, 0:1].permute(0, 2, 1)], dim=-1)
            kv = self.kv(pre_kv.permute(0, 2, 1))
            kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4
            )
            key_padding_mask = torch.cat(
                [
                    key_padding_mask[:, 0:1],
                    key_padding_mask[
                        :, 1 : pooled_kv.shape[-1] * self.kv_pool : self.kv_pool
                    ],
                ],
                dim=1,
            )
            k, v = kv[0], kv[1]
            N_kv = k.shape[2]
        else:
            q = (
                self.q(x)
                .reshape(B, -1, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)
            )
            kv = self.kv(x)
            kv = kv.reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(
                2, 0, 3, 1, 4
            )
            k, v = kv[0], kv[1]
            N_kv = N

        # merge key padding and attention masks
        attn_mask = None
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(B, 1, 1, N_kv).expand(
                -1, self.num_heads, -1, -1
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn += attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return x


class CPE_Block(nn.Module):

    def __init__(self, in_features, kernal_size=3, stride=1, groups=4, cpe_layers=1):
        super(CPE_Block, self).__init__()

        blocks = [
            CPE(in_features, kernal_size, stride, groups) for i in range(cpe_layers)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, dis):
        residual = x
        for block in self.blocks:
            x = block(x, dis)
        return x + residual


class CPE(nn.Module):

    def __init__(self, channels, kernel_size, stride, groups=4, dropout=0):
        super(CPE, self).__init__()
        assert (
            channels % groups == 0
        ), f"dim {channels} should be divided by num_heads {groups}."
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        self.groups = groups

        self.dim = channels
        self.num_heads = groups
        head_dim = channels // groups
        self.scale = head_dim**-0.5

        self.kv_func = nn.Linear(self.dim, self.dim * 2, bias=True)

        self.attn_drop = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.groups),
        )

    def forward(self, x, dis):
        # x: [b, c, t, 1]
        # dis: [b, t, kernel_size, 2]

        b, c, t, w = x.shape

        # b, t, kernel_size, groups
        attn = self.mlp(dis[:, : x.shape[2]])
        # bn, k, groups | n=wt, w=1
        attn = attn.reshape(-1, self.kernel_size, self.groups)
        attn = attn.permute(0, 2, 1).unsqueeze(2)  # bn, groups, 1, k

        k = F.unfold(
            x, (self.kernel_size, 1), 1, ((self.kernel_size - 1) // 2, 0), self.stride
        ).reshape(
            b, c, self.kernel_size, -1
        )  # b, c, k, n

        k = k.permute(0, 3, 2, 1).reshape(-1, self.kernel_size, c)  # bn, k, c
        kv = (
            self.kv_func(k)
            .reshape(
                b * t * w, self.kernel_size, 2, self.num_heads, c // self.num_heads
            )
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]  # bn, groups, k, c'
        # P.S. we find using key doesn't improve performance thus ignore it in the following computations

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b * t * w, 1, c)
        out = x.reshape(b, t, w, c).permute(0, 3, 1, 2)
        return out


class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0, kv_pool=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm_Mask(
                            dim,
                            PoolAttention(
                                dim,
                                heads=heads,
                                dropout=dropout,
                                stage=i,
                                kv_pool=kv_pool,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, mask, dt, dd):
        for attn, ff in self.layers:
            x = attn(x, mask, dt, dd) + x
            x = ff(x) + x
        return x


class TrajFormer(BaseModel):

    def __init__(
        self,
        name,
        c_in=6,  # input_dim
        c_out=4,  # output_dim. In GeoLife, we have 4 modes.
        trans_layers=3,  # number of transformer layers
        n_heads=4,  # number of heads
        token_dim=64,  # token dimension
        kv_pool=1,  # the ratio of pooling
        mlp_dim=256,  # mlp dimension in Transformer
        max_points=100,  # the max points in a trajectory
        cpe_layers=1,  # the number of CPE layers
        device="cuda" if torch.cuda.is_available() else "cpu",  # Add device parameter
    ):
        super(TrajFormer, self).__init__()
        self.name = name
        self.c_in = c_in
        self.c_out = c_out
        self.max_points = max_points
        self.local_layers = cpe_layers
        self.depth = trans_layers
        self.device = device  # Store device

        if cpe_layers > 0:
            self.conv1d = nn.Sequential(
                nn.Conv1d(c_in, token_dim, 1, 1, 0), nn.ReLU(inplace=True)
            )
            self.involution = CPE_Block(token_dim, 9, 1, n_heads, cpe_layers)
        else:
            self.conv1d = nn.Sequential(
                nn.Conv1d(
                    c_in, token_dim, 9, 1, 4
                ),  # kernel is very important, for capturing local
                nn.ReLU(inplace=True),
            )

        if trans_layers > 0:
            self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim))
            trunc_normal_(self.cls_token, std=0.02)
            self.transformer = Transformer(
                token_dim, trans_layers, n_heads, mlp_dim, 0.2, kv_pool
            )

        self.output_layer = nn.Sequential(nn.Linear(token_dim, c_out))

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, m, dis, pad=False):
        if pad:
            inputs, _len = rnn_utils.pad_packed_sequence(x, batch_first=True)
        else:
            inputs = x
            _len = [len(i) for i in x]

        # input: [b, n_time, in_channels]
        inputs = inputs.squeeze()
        dt = inputs[..., 2]  # delta time
        dd = inputs[..., 3]  # delta distance
        b, n_time, in_channels = inputs.shape

        # create mask, considering cls token
        max_len = max(_len)
        mask = torch.BoolTensor(b, max_len + 1).fill_(False).to(self.device)
        for i, s in enumerate(_len):
            if s < max_len:
                mask[i, -(max_len - s.item()) :] = True

        if self.local_layers > 0:
            out = self.conv1d(inputs.permute(0, 2, 1))  # b, token_dim, n_time
            # b, n_time, token_dim
            out = self.involution(out.unsqueeze(-1), dis).squeeze().permute(0, 2, 1)
        else:
            out = self.conv1d(inputs.permute(0, 2, 1)).permute(
                0, 2, 1
            )  # b, n_time, token_dim

        if self.depth > 0:
            cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
            x = torch.cat((cls_tokens, out), dim=1)
            out = self.transformer(x, mask, dt, dd)

        return out

    def forward(self, x, m, dis, pad=False):
        # dis: [b, n_time, kernel_size, 2]
        if pad:
            _, _len = rnn_utils.pad_packed_sequence(x, batch_first=True)
        else:
            inputs = x
            _len = [len(i) for i in x]

        out = self.forward_features(x, m, dis, pad)  # [b, n+1, c]

        if self.depth > 0:
            out = out[:, 0]
        else:
            temp = []
            for i in range(len(out)):
                temp.append(out[i, : _len[i], :].mean(dim=0))
            out = torch.stack(temp, dim=0)

        return self.output_layer(out)  # [b, 4]


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
        from sklearn.preprocessing import LabelEncoder

        self.encoder = LabelEncoder()
        self.labels = data.labels
        y_encoded = self.encoder.fit_transform(self.labels)

        # Create and initialize model
        self.model = TrajFormer(
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
        )

        # Convert data to the format expected by TrajFormer
        x_trajs, masks, distances = self._prepare_data(data)

        # Set up optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        self.model.train()
        n_epochs = kwargs.get("n_epochs", 100)

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            outputs = self.model(x_trajs, masks, distances)
            y_tensor = torch.tensor(y_encoded, dtype=torch.long)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")

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
            probabilities = F.softmax(outputs, dim=1).numpy()

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

    def _prepare_data(self, data: Data):
        """
        Convert data from pactus format to TrajFormer input format

        Returns:
            x_trajs: tensor of trajectory features
            masks: attention masks
            distances: distance matrices for CPE
        """
        # This is a placeholder implementation - actual implementation would depend
        # on the exact data structure and requirements of TrajFormer

        x_trajs = []
        masks = []
        distances = []

        for traj in data.trajs:
            # Process coordinates, time, etc.
            coords = traj.r
            times = traj.t

            # Create feature vector with appropriate dimensions
            features = self._extract_features(coords, times)

            # Create mask and distance matrix
            mask = torch.zeros(self.max_points, dtype=torch.bool)
            if len(features) < self.max_points:
                mask[len(features) :] = True

            # Calculate distances for CPE
            dist_matrix = self._calculate_distances(coords)

            x_trajs.append(features)
            masks.append(mask)
            distances.append(dist_matrix)

        # Convert to tensors and pad sequences if needed
        # This is simplified and would need to be adapted to actual implementation
        return torch.tensor(x_trajs), torch.stack(masks), torch.tensor(distances)

    def _extract_features(self, coords, times):
        """Extract features from trajectory data"""
        # Placeholder - implement based on actual feature requirements
        return np.zeros((len(coords), self.c_in))

    def _calculate_distances(self, coords):
        """Calculate distance matrix for CPE module"""
        # Placeholder - implement based on actual distance calculation needs
        return np.zeros((len(coords), 9, 2))
