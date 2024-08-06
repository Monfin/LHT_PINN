import torch
from torch import nn

from src.data.components.collate import ModelBatch, Coords

from typing import Dict, List

# Register initial conditions, boundary conditions and pde conditions


class PDEPoissonConditions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelBatch) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())


class PDEBurgerCondition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelBatch) -> torch.Tensor:
        # return torch.zeros(size=inputs.time.size())
        return torch.cos(torch.pi * inputs.time) * torch.exp(-1e-1 * inputs.time) 


class PDEOtherCondition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelBatch) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())


class InitialConditions(nn.Module):
    def __init__(self, coords_limits: Dict[str, List[float]]):
        super().__init__()

        self.L = sum([abs(limit) for limit in coords_limits["x"]])

    def forward(self, coords: Coords) -> torch.Tensor:
        return -torch.sin(torch.pi * coords.x / self.L)
        # return torch.exp(-(torch.square(coords.x) + torch.square(coords.y)) / 2)


class BoundaryXYZConditions(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()

        self.temperature = temperature

    def forward(self, inputs: ModelBatch) -> torch.Tensor:
        return torch.ones(size=inputs.time.size()) * self.temperature