import torch
from torch import nn

from src.data.components.collate import SingleForwardState


class PoolingType(nn.Module):
    num_poolings: int = None

    def __init__(self):
        super(PoolingType, self).__init__()


def first_pooling(hidden_state: torch.Tensor, dim: int = 1):
    assert len(hidden_state.size()) == 3, \
        "hidden state size should be (batch_size x num_seq x seq_len)"

    return hidden_state[:, 0, :] if dim == 1 else hidden_state[:, :, 0]


def last_pooling(hidden_state: torch.Tensor, lengths: torch.Tensor, dim: int = 1):
    assert len(hidden_state.size()) == 3, \
        "hidden state size should be (batch_size x num_seq x seq_len)"
    
    if dim == 1:
        hidden_state = hidden_state[torch.arange(hidden_state.size(0)), lengths - 1, :] # (N, L, B)
    elif dim == 2:
        hidden_state = hidden_state[torch.arange(hidden_state.size(0)), :, lengths - 1] # (N, B, L)
    else:
        raise NotImplementedError("dim is not valid, select dim from the <[1, 2]>")

    return hidden_state


def avg_pooling(hidden_state: torch.Tensor, dim: int = 1):
    assert len(hidden_state.size()) in (2, 3), \
        "hidden state size should be (batch_size x num_seq x seq_len) or (num_seq x seq_len)"

    return torch.mean(hidden_state, dim=dim)


def min_pooling(hidden_state: torch.Tensor, dim: int = 1):
    assert len(hidden_state.size()) in (2, 3), \
        "hidden state size should be (batch_size x num_seq x seq_len) or (num_seq x seq_len)"

    return torch.min(hidden_state, dim=dim).values


def max_pooling(hidden_state: torch.Tensor, dim: int = 1):
    assert len(hidden_state.size()) in (2, 3), \
        "hidden state size should be (batch_size x num_seq x seq_len) or (num_seq x seq_len)"

    return torch.max(hidden_state, dim=dim).values


class AvgPooling(PoolingType):
    def __init__(self, dim: int = 1):
        super(AvgPooling, self).__init__()

        self.dim = dim

        self.num_poolings = 1

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            avg_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled


class LastPooling(PoolingType):
    def __init__(self, dim: int = 1):
        super(LastPooling, self).__init__()

        self.dim = dim

        self.num_poolings = 1

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            last_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled


class FirstLastPoolings(PoolingType):
    def __init__(self, dim: int = 1):
        super(FirstLastPoolings, self).__init__()

        self.dim = dim

        self.num_poolings = 2

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            first_pooling(hidden_state, self.dim),
            last_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled
    

class FirstLastAvgPoolings(PoolingType):
    def __init__(self, dim: int = 1):
        super(FirstLastAvgPoolings, self).__init__()

        self.dim = dim

        self.num_poolings = 3

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            first_pooling(hidden_state, self.dim),
            last_pooling(hidden_state, self.dim),
            avg_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled


class MinMaxPoolings(PoolingType):
    def __init__(self, dim: int = 1):
        super(MinMaxPoolings, self).__init__()

        self.dim = dim

        self.num_poolings = 2

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            min_pooling(hidden_state, self.dim),
            max_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled


class MinMaxAvgPoolings(PoolingType):
    def __init__(self, dim: int = 1):
        super(MinMaxAvgPoolings, self).__init__()

        self.dim = dim

        self.num_poolings = 3

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            min_pooling(hidden_state, self.dim),
            max_pooling(hidden_state, self.dim),
            avg_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled


class AllPoolings(PoolingType):
    def __init__(self, dim: int = 1):
        super(AllPoolings, self).__init__()

        self.dim = dim

        self.num_poolings = 5

    def forward(self, hidden_state: torch.Tensor):
        pooled_results = [
            first_pooling(hidden_state, self.dim),
            last_pooling(hidden_state, self.dim),
            min_pooling(hidden_state, self.dim),
            max_pooling(hidden_state, self.dim),
            avg_pooling(hidden_state, self.dim)
        ]
        hidden_state_pooled = torch.stack(pooled_results, dim=self.dim)

        return hidden_state_pooled


POOLING_MAPPING = {
    "all": AllPoolings,
    "avg": AvgPooling,
    "min_max": MinMaxPoolings,
    "min_max_avg": MinMaxAvgPoolings,
    "last": LastPooling,
    "first_last": FirstLastPoolings,
    "first_last_avg": FirstLastAvgPoolings
}

    
class ConvPooling(nn.Module):
    def __init__(
            self, 
            emb_dim: int,
            pooling_type: str = "all", 
            use_batch_norm: bool = False,
            dim: int = 1
        ) -> None:
        super(ConvPooling, self).__init__()

        self.dim = dim

        pooling_types = list(POOLING_MAPPING.keys())
        assert pooling_type in pooling_types, \
            f"You should specify pooling type from {pooling_types}, not {pooling_type}"
        
        self.pooling_layer = POOLING_MAPPING[pooling_type](dim=dim)

        self.agg_layer = nn.Conv1d(
            in_channels=self.pooling_layer.num_poolings, 
            out_channels=1, 
            kernel_size=self.pooling_layer.num_poolings, 
            padding="same",
            bias=False
        )

        self.batch_norm = nn.BatchNorm1d(emb_dim) if use_batch_norm else nn.Identity()

        
    def forward(self, hidden_state: SingleForwardState) -> SingleForwardState:

        x = self.pooling_layer(hidden_state.sequences)
        x = self.agg_layer(x).squeeze(self.dim)
        x = self.batch_norm(x)

        return SingleForwardState(
            sequences=x
        )