import torch 
from torch import nn

from typing import List

from src.data.components.collate import ModelInput, ModelOutput


class PDESimpleNN(nn.Module):
    def __init__(
            self, 
            layers: List[nn.Module], 
            embedding_dim: int = 4
        ) -> None:
        super().__init__()

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: ModelInput) -> ModelOutput:

        state = self.layers(inputs)

        return state