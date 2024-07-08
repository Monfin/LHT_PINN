import torch
from torch import nn

from src.data.components.collate import SingleForwardState, ModelInput


class MainEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 8,
            dropout_inputs: float = 0.3,
            num_coords: int = 1
        ) -> None:
        super(MainEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout_inputs)

        self.branched_linear_block_xyz = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, embedding_dim),
                    nn.Tanh()
                ) for _ in range(num_coords)
            ]
        )

        self.linear_block_t = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh()
        )

        self.out_linear_block = nn.Linear(
            embedding_dim * (num_coords + 1), embedding_dim
        )
        

    def forward(self, inputs: ModelInput) -> SingleForwardState:

        coords = [coord for coord in inputs.coords if coord is not None]

        coords_emb = torch.concatenate(
            [
                head(coord) for coord, head in zip(coords, self.branched_linear_block_xyz)
            ], dim=1
        )

        time_emb = self.linear_block_t(inputs.time)

        # emb = torch.stack([coords_emb, time_emb], dim=-1) # (batch_size, emb_dim, num_coords + 1)
        emb = torch.concatenate([coords_emb, time_emb], dim=-1)

        x = self.dropout(emb)
        x = self.out_linear_block(x)

        return SingleForwardState(
            sequences=x
        )