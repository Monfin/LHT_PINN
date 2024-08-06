import torch
from torch import nn

from src.data.components.collate import ModelBatch


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
            nn.Linear(4, embedding_dim),
            nn.Tanh()
        )

        self.out_linear_block = nn.Sequential(
            nn.Linear(
                embedding_dim * (num_coords + 1), embedding_dim
            ),
            nn.BatchNorm1d(embedding_dim)
        )
        self.res_bn = nn.BatchNorm1d(embedding_dim)
        

    def forward(self, inputs: ModelBatch) -> torch.Tensor:
        coords_emb = torch.concatenate(
            [
                head(coord) for coord, head in zip(inputs.coords, self.branched_linear_block_xyz)
            ], dim=1
        )

        time_emb = self.linear_block_t(
            torch.concatenate(
                (
                    inputs.time,
                    torch.log(inputs.time + 1e-6), 
                    torch.cos(inputs.time),
                    torch.sin(inputs.time)
                ), dim=-1
            )
        )

        coords_time_emb = torch.concatenate([coords_emb, time_emb], dim=-1)

        # emb = torch.stack([coords_emb, time_emb], dim=-1) # (batch_size, emb_dim, num_coords + 1)
        emb = self.out_linear_block(self.dropout(coords_time_emb))
        res_emb = self.res_bn(coords_emb * time_emb)

        x = emb + res_emb

        return x