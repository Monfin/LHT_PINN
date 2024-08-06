import torch 
from torch import nn

from typing import List

from src.data.components.collate import ModelBatch, Coords


class SimplePINN(nn.Module):
    def __init__(
            self, 
            layers: List[nn.Module], 
            embedding_dim: int = 4
        ) -> None:
        super().__init__()

        self.layers = nn.Sequential(*layers)
        

    def forward(self, inputs: ModelBatch) -> torch.Tensor:

        state = self.layers(inputs)

        return state
    

# for tracing
class TracedSimplePINN(SimplePINN):
    def __init__(self, **kwargs):
        super(TracedSimplePINN, self).__init__(**kwargs)


    def forward(self, coords: Coords, time: torch.Tensor) -> torch.Tensor:

        inputs = ModelBatch(coords, time)

        return super().forward(inputs)