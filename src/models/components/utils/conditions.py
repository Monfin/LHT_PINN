import torch
from torch import nn

from src.data.components.collate import ModelInput, Coords

# Register initial conditions, boundary conditions and pde conditions


class PDEPoissonConditions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())


class PDEBurgerCondition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())


class PDEOtherCondition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())


class InitialConditions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords: Coords) -> torch.Tensor:
        return torch.exp(-(torch.square(coords.x) + torch.square(coords.y)) / 2)


class BoundaryXYZConditions(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()

        self.temperature = temperature


    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return torch.ones(size=inputs.time.size()) * self.temperature