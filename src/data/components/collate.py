from dataclasses import dataclass

import torch

from typing import List, Dict

from collections import namedtuple

from abc import ABC


# dataclass - abstract class for model batches (data type)
# returns class with __init__, __repr__ and other

_coords = namedtuple(typename="coords", field_names=["x", "y", "z"])
class Coords(_coords):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor

    def __new__(
        cls, 
        x: torch.Tensor = torch.tensor([]),
        y: torch.Tensor = torch.tensor([]),
        z: torch.Tensor = torch.tensor([])
    ):
        return super().__new__(cls, x=x, y=y, z=z)


_model_input = namedtuple(typename="SpatialTemporalDomain", field_names=["coords", "time"])
class ModelBatch(_model_input):
    coords: Coords
    time: torch.Tensor

    def __new__(
        cls,
        coords: Coords = Coords(),
        time: torch.Tensor = torch.tensor([])
    ):
        return super().__new__(cls, coords=coords, time=time)


_model_output = namedtuple(typename="SpatialTemporalDomainSolution", field_names=["model_batch", "solution"])
class ModelOutput(_model_output):
    model_batch: ModelBatch
    solution: torch.Tensor

    def __new__(
        cls,
        model_batch: ModelBatch = ModelBatch(),
        solution: torch.Tensor = torch.tensor([])
    ):
        return super().__new__(cls, model_batch=model_batch, solution=solution)


class Collator(ABC):
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict]) -> ModelBatch:
        pass


class BaseCollator(Collator):
    def __call__(self, batch: List[Dict]) -> ModelBatch:
        coords = dict()

        _item = batch[0]

        for key in _item["coords"].keys():
            coords[key] = torch.stack([item["coords"][key] for item in batch], dim=0)
            coords[key].requires_grad_(True)


        if _item["time"].__len__() > 0:
            time = torch.stack([item["time"] for item in batch], dim=0)
            time.requires_grad_(True)
        else:
            time = torch.tensor([])

        return ModelBatch(
            coords=Coords(**coords), 
            time=time
        )
    

@dataclass
class BaseCollator2D:
    seq_len: int = 128

    def __call__(self, batch: List[Dict]) -> ModelBatch:
        batch_size = len(batch)

        coords = dict()
        
        _item = batch[0]

        for key in _item["coords"].keys():
            coords[key] = torch.stack([item[key] for item in batch], dim=0).view(batch_size * self.seq_len, 1)
            coords[key].requires_grad_(True)


        if _item["time"].__len__() > 0:
            time = torch.stack([item["time"] for item in batch], dim=0).view(batch_size * self.seq_len, 1)
            time.requires_grad_(True)
        else:
            time = torch.tensor([])


        return ModelBatch(
            coords=Coords(**coords), 
            time=time
        )