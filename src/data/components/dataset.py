import torch


class SpatialXTemporalDomain2D(torch.utils.data.Dataset):
    def __init__(
            self, 
            n_samples: int = 10_000, 
            seq_len: int = 128,
            xmin: float = 0.0, 
            xmax: float = 1.0, 
            tmin: float = 0.0, 
            tmax: float = 1.0,
            noise: float = 0.0
        ) -> None:

        self.data_size = n_samples

        self.nrof_bins = self.data_size // seq_len

        _X: torch.Tensor = torch.empty(size=(n_samples, 1))
        _X.uniform_(xmin, xmax)

        _X = torch.sort(_X.ravel()).values
        
        self.X = torch.stack(
            [
                _X[bin * seq_len:bin * seq_len + seq_len][torch.randperm(seq_len)] for bin in range(self.nrof_bins)
            ]
        )[torch.randperm(self.nrof_bins)]

        self.X += noise * torch.randn(size=self.X.size())


        _T: torch.Tensor = torch.empty(size=(n_samples, 1))
        _T.uniform_(tmin, tmax)

        _T = torch.sort(_T.ravel()).values
        
        self.T = torch.stack(
            [
                _T[bin * seq_len:bin * seq_len + seq_len][torch.randperm(seq_len)] for bin in range(self.nrof_bins)
            ]
        )[torch.randperm(self.nrof_bins)]

        self.T += noise * torch.randn(size=self.T.size())


    def __getitem__(self, idx: int):
        return {
            "coords": {
                "x": self.X[idx]
            },
            "time": self.T[idx]
        }

    def __len__(self):
        return self.nrof_bins


class SpatialXTemporalDomain(torch.utils.data.Dataset):
    def __init__(
            self, 
            n_samples: int = 10_000, 
            xmin: float = 0.0, 
            xmax: float = 1.0, 
            tmin: float = 0.0, 
            tmax: float = 1.0,
            noise: float = 0.0
        ) -> None:

        self.data_size = n_samples

        self.X: torch.Tensor = torch.empty(size=(n_samples, 1))
        self.X.uniform_(xmin, xmax)
        self.X += noise * torch.randn(size=(self.X.size()))

        self.T: torch.Tensor = torch.empty(size=(n_samples, 1))
        self.T.uniform_(tmin, tmax)
        self.T += noise * torch.randn(size=(self.T.size()))

    def __getitem__(self, idx: int):
        return {
            "coords": {
                "x": self.X[idx]
            },
            "time": self.T[idx]
        }

    def __len__(self):
        return self.data_size


class SpatialXYTemporalDomain(torch.utils.data.Dataset):
    def __init__(
            self, 
            n_samples: int = 10_000, 
            xmin: float = 0.0, 
            xmax: float = 1.0, 
            ymin: float = 0.0, 
            ymax: float = 1.0,
            noise: float = 0.0
        ) -> None:

        self.data_size = n_samples

        self.X: torch.Tensor = torch.empty(size=(n_samples, 1))
        self.X.uniform_(xmin, xmax)
        self.X += noise * torch.randn(size=(self.X.size()))

        self.Y: torch.Tensor = torch.empty(size=(n_samples, 1))
        self.Y.uniform_(ymin, ymax)
        self.Y += noise * torch.randn(size=(self.Y.size()))

    def __getitem__(self, idx: int):
        return {
            "coords": {
                "x": self.X[idx],
                "y": self.Y[idx]
            },
            "time": None
        }

    def __len__(self):
        return self.data_size