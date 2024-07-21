from dataclasses import dataclass

import torch

from typing import List, Dict, Optional

# from collections import namedtuple


# abstract class for model batches (data type)
# returns class with __init__, __repr__ and other
@dataclass
class ModelBatch:
    coords: Optional[torch.Tensor]
    time: Optional[torch.Tensor]


@dataclass
class SingleForwardState:
    sequences: Optional[torch.Tensor]


@dataclass
class TwoBranchForwardState:
    main_seq: SingleForwardState
    aggregates: Optional[torch.Tensor]


@dataclass
class ModelOutput:
    representations: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]


# coords = namedtuple(typename="Coords", field_names=["x", "y", "z"])

@dataclass
class Coords:
    x: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    z: Optional[torch.Tensor] = None

    def __iter__(self):
        return iter([self.x, self.y, self.z])


# model_input = namedtuple(typename="inputs", field_names=["x", "y", "z", "time"])

# @dataclass
# class ModelInput(model_input):
#     x: Optional[torch.Tensor] = None
#     y: Optional[torch.Tensor] = None
#     z: Optional[torch.Tensor] = None
#     time: Optional[torch.Tensor] = None

#     def __iter__(self):
#         return iter([self.x, self.y, self.z])

@dataclass
class ModelInput:
    coords: Coords
    time: Optional[torch.Tensor]


class BaseCollator:
    def __init__(self, with_coords: bool = True, with_time: bool = True):
        self.with_coords = with_coords
        self.with_time = with_time

    def __call__(self, batch: List[Dict]) -> ModelBatch: # ModelInput
        if self.with_coords:
            coords_keys = ["x", "y", "z"]

            coords = dict().fromkeys(coords_keys)

            # for key in coords_keys:
            #     coords[key] = torch.stack([item[key] for item in batch], dim=0)
            for key in batch[0]["coords"].keys():
                coords[key] = torch.stack([item["coords"][key] for item in batch], dim=0)


                coords[key].requires_grad_(True)
        else:
            coords = None

        if self.with_time:
            time = torch.stack([item["time"] for item in batch], dim=0) 

            time.requires_grad_(True)
        else:
            time = None


        return ModelBatch(
            coords=Coords(**coords), 
            time=time
        )

        # return ModelInput(
        #     **coords, 
        #     time=time
        # )
    

class BaseCollator2D:
    def __init__(self, with_coords: bool = True, with_time: bool = True, seq_len: int = 128):
        self.with_coords = with_coords
        self.with_time = with_time
        self.seq_len = seq_len

    def __call__(self, batch: List[Dict]) -> ModelInput:
        batch_size = len(batch)

        if self.with_coords:
            coords_keys = ["x", "y", "z"]

            coords = dict().fromkeys(coords_keys)

            for key in batch[0].keys():
                coords[key] = torch.stack([item[key] for item in batch], dim=0).view(batch_size * self.seq_len, 1)

                coords[key].requires_grad_(True)
        else:
            coords = None

        if self.with_time:
            time = torch.stack([item["time"] for item in batch], dim=0).view(batch_size * self.seq_len, 1)

            time.requires_grad_(True)
        else:
            time = None

        return ModelInput(
            **coords, 
            time=time
        )