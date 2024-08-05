import torch

from typing import List, Dict


class SpatialTemporalDomain2D(torch.utils.data.Dataset):
    def __init__(
            self, 
            coords_limits: Dict[str, List[float]],
            time_limits: List[float],
            n_samples: int = 10_000, 
            seq_len: int = 128,
            noise: float = 0.0
        ) -> None:

        self.data_size = n_samples

        self.nrof_bins = self.data_size // seq_len

        coords = dict()

        for coord_key, coord_limits in coords_limits.items():
            _X: torch.Tensor = torch.empty(size=(n_samples, ))
            _X.uniform_(coord_limits[0], coord_limits[1])

            _X = torch.sort(_X).values
            
            X = torch.stack(
                [
                    _X[bin * seq_len:bin * seq_len + seq_len][torch.randperm(seq_len)] for bin in range(self.nrof_bins)
                ]
            )[torch.randperm(self.nrof_bins)]

            coords[coord_key] = X

        self.coords = coords


        _T: torch.Tensor = torch.empty(size=(n_samples, ))
        _T.uniform_(time_limits[0], time_limits[1])

        _T = torch.sort(_T).values
        
        self.T = torch.stack(
            [
                _T[bin * seq_len:bin * seq_len + seq_len][torch.randperm(seq_len)] for bin in range(self.nrof_bins)
            ]
        )[torch.randperm(self.nrof_bins)]


    def __getitem__(self, idx: int):
        return {
            "coords": {
                coord_key: coord[idx] for coord_key, coord in self.coords.items()
            },
            "time": self.T[idx]
        }

    def __len__(self):
        return self.nrof_bins


class SpatialTemporalDomain(torch.utils.data.Dataset):
    def __init__(
            self, 
            coords_limits: Dict[str, List[float]],
            time_limits: Dict[str, float],
            n_samples: int = 10_000
        ) -> None:

        self.data_size = n_samples

        coords = dict()

        for coord_key, coord_limits in coords_limits.items():
            X: torch.Tensor = torch.empty(size=(n_samples, 1))
            X.uniform_(coord_limits[0], coord_limits[1])

            coords[coord_key] = X

        self.coords = coords

        self.T: torch.Tensor = torch.empty(size=(n_samples, 1))
        self.T.uniform_(time_limits[0], time_limits[1])


    def __getitem__(self, idx: int):
        return {
            "coords": {
                coord_key: coord[idx] for coord_key, coord in self.coords.items()
            },
            "time": self.T[idx]
        }

    def __len__(self):
        return self.data_size