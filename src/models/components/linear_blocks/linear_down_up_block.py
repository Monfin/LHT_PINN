import torch
from torch import nn
from typing import List

from src.data.components.collate import SingleForwardState, ModelOutput

ACTIVATION_TYPE_MAPPING = {
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "none": nn.Identity
}

def init_linear_block_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(layer.bias)

class LinearDownUpBlock(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int = 1, 
            down: bool = True,
            num_layers: int = 3, 
            dropout_rate: float = 0.0, 
            activation_type: str = "tanh",
            use_batch_norm: bool = False,
            bias: bool = True
        ) -> None:
        super(LinearDownUpBlock, self).__init__()

        self.in_features = in_features

        self.dropout = nn.Dropout(p=dropout_rate)

        if activation_type is None:
            self.act = ACTIVATION_TYPE_MAPPING["tanh"]
        elif activation_type in ACTIVATION_TYPE_MAPPING.keys():
            self.act = ACTIVATION_TYPE_MAPPING[activation_type]
        else: 
            NotImplementedError(f"activation_type must be in <{list(ACTIVATION_TYPE_MAPPING.keys())}>")

        if use_batch_norm:
            self.layer_norm = nn.BatchNorm1d
        else:
            self.layer_norm = nn.LayerNorm

        features_dim = lambda n_dim, k: n_dim // (2 ** k) if down else n_dim * (2 ** k)

        self.linear_block = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        nn.Linear(
                            features_dim(in_features, i), 
                            features_dim(in_features, i + 1), 
                            bias
                        ),
                        self.layer_norm(features_dim(in_features, i + 1)),
                        self.act()
                    ]
                ) for i in range(num_layers)
            ]
        )
        

        self.out_block = nn.Linear(features_dim(in_features, num_layers), out_features)

        self.cls_layers = nn.Sequential(
            self.dropout,
            self.linear_block,
            self.out_block,
            self.act()
        )

        # weights init
        self.cls_layers.apply(init_linear_block_weights)


    def forward(self, x: SingleForwardState) -> ModelOutput:
        logits = self.cls_layers(x.sequences)

        return ModelOutput(
            representations=x.sequences,
            logits=logits
        )