import torch
from torch import nn

ACTIVATION_TYPE_MAPPING = {
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "none": nn.Identity
}

class LinearDownUpBlock(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int = 1, 
            reduce: bool = False,
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

        features_dim = lambda n_dim, k: n_dim // (2 ** k) if down else n_dim * (2 ** k)

        self.linear_block = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        nn.Linear(
                            features_dim(in_features, i) if reduce else in_features, 
                            features_dim(in_features, i + 1) if reduce else in_features, 
                            bias
                        ),
                        nn.BatchNorm1d(features_dim(in_features, i + 1) if reduce else in_features) if use_batch_norm else nn.Identity(),
                        self.act()
                    ]
                ) for i in range(num_layers)
            ]
        )
        

        self.out_block = nn.Linear(features_dim(in_features, num_layers) if reduce else in_features, out_features)

        self.cls_layers = nn.Sequential(
            self.dropout,
            self.linear_block,
            self.out_block
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.cls_layers(x)

        return logits