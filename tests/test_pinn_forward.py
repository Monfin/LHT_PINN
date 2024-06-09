import rootutils

rootutils.setup_root(search_from=__file__, indicator=".project-root", pythonpath=True)

from src.models.pde_lit_model_xt import PDELitModule
from src.models.components.pde_nn import PDESimpleNN
from src.models.components.linear_blocks.linear_down_up_block import LinearDownUpBlock

import torch
from torch import nn

from src.data.components.dataset import SpatialXTemporalDomain
from src.data.components.collate import BaseCollator, ModelInput, SingleForwardState, Coords


BATCH_SIZE = 128 
EMB_DIM = 8

def get_sample():
    dataset = SpatialXTemporalDomain(n_samples=100_000, xmin=-3, xmax=3, tmin=0, tmax=5)

    train, valid = torch.utils.data.random_split(
        dataset, [0.9, 0.1]
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        collate_fn=BaseCollator()
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        collate_fn=BaseCollator()
    )

    return next(iter(train_loader))


sample = get_sample()


class PDEBurgerCondition(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())


class InitialConditions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, coords: Coords) -> torch.Tensor:
        return torch.exp(-torch.square(coords.x) / 2)
        # return torch.zeros(coords.x.size())


class BoundaryXYZConditions(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: ModelInput) -> torch.Tensor:
        return torch.zeros(size=inputs.time.size())
    

class MainEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_sequence_dim: int = 8,
            embedding_features_dim: int = 0,
            dropout_inputs: float = 0.3,
            num_coords: bool = False,
            with_time: bool = False
        ) -> None:
        super(MainEncoderLayer, self).__init__()

        self.dropout = nn.Dropout(p=dropout_inputs)

        self.branched_linear_block_xyz = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, embedding_sequence_dim),
                    nn.Tanh()
                ) for _ in range(num_coords)
            ]
        )

        self.linear_block_t = nn.Sequential(
            nn.Linear(1, embedding_sequence_dim),
            nn.Tanh()
        ) if with_time else None

        if embedding_features_dim > 0:
            in_features = num_coords + 1 if with_time else num_coords

            self.out_linear_block = nn.Linear(
                in_features, embedding_features_dim
            )
        else:
            self.out_linear_block = nn.Identity()


    def forward(self, inputs: ModelInput) -> SingleForwardState:

        coords = [coord for coord in inputs.coords if coord is not None]

        coords_emb = torch.concatenate(
            [
                head(coord) for coord, head in zip(coords, self.branched_linear_block_xyz)
            ], dim=1
        )

        time_emb = self.linear_block_t(inputs.time)

        # emb = torch.stack([coords_emb, time_emb], dim=-1) # (batch_size, emb_dim, num_coords + 1)
        emb = torch.concatenate([coords_emb, time_emb], dim=-1)

        x = self.dropout(emb)
        # x = self.out_linear_block(x)

        return SingleForwardState(
            sequences=x
        )

model = PDESimpleNN(
    layers=[
        MainEncoderLayer(
            embedding_sequence_dim=EMB_DIM, 
            embedding_features_dim=1, 
            dropout_inputs=0.3, 
            num_coords=1, 
            with_time=True
        ),
        LinearDownUpBlock(
            in_features=EMB_DIM * 2,
            out_features=1,
            down=True,
            num_layers=1,
            dropout_rate=0.0,
            activation_type="tanh",
            use_batch_norm=False
        )
    ]
)

lit_module = PDELitModule(
    net=model,
    train_batch_size=BATCH_SIZE,
    val_batch_size=BATCH_SIZE,
    conditional_loss="val/loss",
    condition_names=["pdec", "ic", "bc_lower", "bc_upper"],
    num_coords=1,
    pdec=PDEBurgerCondition(),
    ic=[InitialConditions()],
    bc_limits=[-3.0, 3.0],
    bc=[BoundaryXYZConditions(), BoundaryXYZConditions()],
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
)


print(lit_module.model_step(sample))