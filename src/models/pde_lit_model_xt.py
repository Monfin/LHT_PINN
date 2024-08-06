import torch
from torch import nn

from typing import Dict

import torch
from torch import nn

from src.models.components.pde_nn import SimplePINN, TracedSimplePINN

from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import RelativeSquaredError

from src.data.components.collate import ModelBatch, ModelOutput, Coords

from typing import Dict, Tuple, List, Union

import lightning as L

import logging
log = logging.getLogger(__name__)


METRICS_MAPPING = {
    "rse": RelativeSquaredError
}


class PDELitModule(L.LightningModule):
    def __init__(
            self, 
            net: SimplePINN,

            train_batch_size: int,
            val_batch_size: int,

            condition_names: List[str],

            # metric_names: List[str],
            conditional_loss: str,

            optimizer: torch.optim.Optimizer,

            conditions: Dict[str, Union[nn.Module, List[nn.Module]]],

            num_coords: int = 1,

            alpha: float = 1.0,
            beta: float = 1.0,
            nu: float = 0.0,

            bc_limits: Dict[str, List[float]] = {"x": [0.0]},

            scheduler: torch.optim.lr_scheduler = None,

            compile: bool = False
        ) -> None:

        super(PDELitModule, self).__init__()

        self.save_hyperparameters()

        self.conditional_loss = conditional_loss

        self.net = net

        self.pdec = conditions["pdec"]
        self.other_pdec = conditions["other_pdec"]
        self.ic = conditions["ic"]
        self.bc = conditions["bc"]

        self.criterion = nn.ModuleList(
            [
                nn.MSELoss() for _ in range(num_coords)
            ]
        )

        if len(self.other_pdec) > 0:
            self.other_pdec_criterion = nn.ModuleList(
                [
                    nn.MSELoss() for _ in range(len(self.other_pdec))
                ]
            )
        else:
            self.other_pdec_criterion = None

        
        if len(self.ic) > 0:
            # self.ic_criterion = nn.ModuleList(
            #     [
            #         nn.ModuleList(
            #             [
            #                 nn.MSELoss() for _ in range(num_coords)
            #             ]
            #         ) for _ in range(len(self.ic))
            #     ]
            # )
            self.ic_criterion = nn.ModuleList(
                [
                    nn.MSELoss()  for _ in range(len(self.ic))
                ]
            )
        else:
            self.ic_criterion = None

        
        self.bc_limits = bc_limits

        if len(self.bc) > 0:
            self.bc_criterion = nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            nn.MSELoss() for _ in range(num_coords)
                        ]
                    ) for _ in range(len(self.bc))
                ]
            )
        else:
            self.bc_criterion = None
        
        self.condition_names = condition_names

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_branched_loss = nn.ModuleDict({key: MeanMetric() for key in self.condition_names})


        self.val_loss = MeanMetric()
        self.val_branched_loss = nn.ModuleDict({key: MeanMetric() for key in self.condition_names})

        self.monitor_loss = self.val_loss
        
        self.reset_valid_scoring()

        # for tracking best so far validation gini
        self.val_best_loss = MinMetric()


    def reset_valid_scoring(self):
        self.val_scoring_losses = {
            "loss": list(),
            "branched_loss": {
                branch: list() for branch in self.condition_names
            }
        }


    def forward(self, inputs: ModelBatch) -> ModelOutput:
        return self.net(*inputs) if self.hparams.compile else self.net(inputs)
    

    def pde(self, u_t, u, u_x, u_xx) -> torch.Tensor:
        # u_x == u1_x or u2_x

        time_part = self.hparams.alpha * u_t

        grad_part = self.hparams.beta * (
            torch.concatenate(
                [
                    ui * ui_x for ui, ui_x in zip(u, u_x)
                ], dim=-1
            ).sum(dim=-1, keepdim=True)
        )

        viscous_part = self.hparams.nu * (
            torch.concatenate(u_xx, dim=-1)
        ).sum(dim=-1, keepdim=True)

        # Burger's equation
        return time_part + grad_part - viscous_part
    

    def other_pde(self, u) -> torch.Tensor:
        # Continuity equation
        return (
            torch.concatenate(u, dim=-1)
        ).sum(dim=-1, keepdim=True)


    # @torch.jit.script
    def derivative(self, u: torch.Tensor, x: torch.Tensor, n: int = 1):
        derivatives = list()
        
        du_n = u

        for _ in range(n):
            du_n = torch.autograd.grad(
                outputs=du_n, 
                inputs=x, 
                grad_outputs=torch.ones_like(u), 
                create_graph=True, 
                retain_graph=True, 
                allow_unused=True
            )[0]

            derivatives.append(du_n)
        
        return derivatives


    def pde_forward(self, inputs: ModelBatch): # PDE Loss
        outputs = self.forward(inputs) # (batch_size x num_coords) ~ [u1(x1, x2; t), u2(x1, x2; t)]
        outputs = ModelOutput(model_batch=inputs, solution=outputs)

        if outputs.solution.requires_grad:
            
            u = [
                tensor.unsqueeze(dim=-1) for tensor in outputs.solution.unbind(dim=-1)
            ] # list(batch_size x num_coords)

            loss = 0.0
            other_grads = list()

            _coords = {
                key: coord for key, coord in inputs.coords._asdict().items() if coord.__len__() > 0
            }.items()

            for idx, ui in enumerate(u):
                u_t = self.derivative(ui, inputs.time, 1)[0]

                u_grads = [dict(), dict()]

                # compute ui_x, ui_xx
                for key, coord in _coords:
                    if coord is not None:
                        u_grads[0][key], u_grads[1][key] = self.derivative(ui, coord, 2)

                pde = self.pde(
                    u_t=u_t, # ui_t
                    u=u, # ui
                    u_x=list(u_grads[0].values()), # ui_x, ui_y
                    u_xx=list(u_grads[1].values()) # ui_xx, ui_yy
                ) # TODO universal

                loss += self.criterion[idx](pde, self.pdec(inputs))

                if self.other_pdec_criterion is not None:
                    other_grads.append(u_grads[0][list(u_grads[0].keys())[idx]]) # diag dix ~ ui


            if self.other_pdec_criterion is not None:
                for criterion, condition in zip(self.other_pdec_criterion, self.other_pdec):

                    other_pde = self.other_pde(u=other_grads)

                    loss += criterion(other_pde, condition(inputs))

            return outputs, pde, loss
        else:
            return outputs, torch.tensor(0.), torch.tensor(0.)

        
    def ic_forward(self, inputs: ModelBatch): # IC Loss
        ic_multi_u = list()
        ic_multi_loss = 0.0

        for criterion, condition in zip(self.ic_criterion, self.ic):

            ic_inputs = ModelBatch(
                coords=inputs.coords,
                time=torch.zeros(size=(inputs.time.size()))
            )

            ic_multi_u.append(self.forward(ic_inputs))
            ic_multi_u[-1] = ModelOutput(model_batch=ic_inputs, solution=ic_multi_u[-1])

            # ic_u = [
            #     tensor.unsqueeze(dim=-1) for tensor in ic_multi_u[-1].logits.unbind(dim=-1)
            # ]

            # for idx, ic_ui in enumerate(ic_u):
            #     ic_multi_loss += criterion[idx](ic_ui, condition(inputs.coords))

            # ic_u_vector = torch.sqrt(torch.sum(torch.square(ic_multi_u[-1].logits), dim=-1, keepdim=True))

            ic_u_vector = ic_multi_u[-1].solution

            ic_multi_loss += criterion(ic_u_vector, condition(inputs.coords))

        return ic_multi_u, ic_multi_loss
    
    
    def bc_forward(self, inputs: ModelBatch): # BC Loss
        _coords = {
            key: coord for key, coord in inputs.coords._asdict().items() if coord.__len__() > 0
        }.items()

        bc_multi_u = list()
        bc_multi_loss = list()

        for index, (branch_criterion, condition) in enumerate(zip(self.bc_criterion, self.bc)):
            bc_u = list()
            bc_loss = 0.0

            for (key, coord), criterion in zip(_coords, branch_criterion):
                if coord is not None:
                    bc_inputs = ModelBatch(
                        coords=Coords(
                            **{
                                nested_key: nested_coord \
                                        if nested_key != key else \
                                            self.bc_limits[key][index] * torch.ones(size=nested_coord.size()) \
                                    for nested_key, nested_coord in _coords
                            }
                        ),
                        time=inputs.time
                    )

                    bc_u.append(self.forward(bc_inputs))
                    bc_u[-1] = ModelOutput(model_batch=bc_inputs, solution=bc_u[-1])

                    # bc_u_vector = torch.sqrt(torch.sum(torch.square(bc_u[-1].logits), dim=-1, keepdim=True))

                    bc_u_vector = bc_u[-1].solution

                    bc_loss += criterion(bc_u_vector, condition(bc_inputs))

            bc_multi_u.append(bc_u)
            bc_multi_loss.append(bc_loss)

        return bc_multi_u, bc_multi_loss
    

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.monitor_loss.reset()
    
        
    def model_step(self, inputs: ModelBatch) -> Tuple[torch.Tensor]:

        u, pde, loss = self.pde_forward(inputs)

        loss_pde = loss.clone().detach()

        branched_loss = [loss_pde]

        if self.ic_criterion is not None:
            _, loss_ic = self.ic_forward(inputs)

            loss += loss_ic

            branched_loss.append(loss_ic.clone().detach())

        if self.bc_criterion is not None:
            _, loss_bc = self.bc_forward(inputs)

            loss += sum(loss_bc)

            branched_loss.extend([loss_item.clone().detach() for loss_item in loss_bc])

        branched_loss = dict(
            zip(
                self.condition_names, 
                branched_loss
            )
        )

        return loss, branched_loss, u, pde


    def training_step(self, batch: ModelBatch, batch_idx: int, dataloader_idx: int = 0) -> Dict:
        """
        :param enable_graph: If True, will not auto detach the graph. 
        """

        loss, branched_loss, _, _ = self.model_step(batch)

        self.train_loss(loss)

        self.log(
            "train/loss", 
            self.train_loss, 
            batch_size=self.hparams.train_batch_size,
            on_step=True, on_epoch=True, prog_bar=True, sync_dist=False
        )

        for branch, loss_item in branched_loss.items():
            self.train_branched_loss[branch](loss_item)

            self.log(
                f"train/{branch}", 
                self.train_branched_loss[branch], 
                batch_size=self.hparams.train_batch_size,
                on_step=True, on_epoch=True, prog_bar=True, sync_dist=False
            )

        return {"loss": loss}


    def validation_step(self, batch: ModelBatch, batch_idx: int, dataloader_idx: int = 0) -> None:
        loss, branched_loss, _, _ = self.model_step(batch)

        self.val_scoring_losses["loss"].append(loss)

        for branch in self.condition_names:
            self.val_scoring_losses["branched_loss"][branch].append(branched_loss[branch])

        return None
    

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        loss = torch.stack(self.val_scoring_losses["loss"]).mean()

        self.val_loss(loss)

        self.log(
            "val/loss", 
            self.val_loss, 
            on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
        )

        branched_loss = dict().fromkeys(self.condition_names)

        for branch in self.condition_names:
            branched_loss[branch] = torch.stack(
                self.val_scoring_losses["branched_loss"][branch]
            ).mean()

            self.val_branched_loss[branch](branched_loss[branch])

            self.log(
                f"val/loss_{branch}",
                self.val_branched_loss[branch],
                on_step=False, on_epoch=True, prog_bar=True, sync_dist=True
            )

        self.val_best_loss(self.monitor_loss.compute())

        self.log(
            "val/loss_best", 
            self.val_best_loss.compute(), 
            sync_dist=True, prog_bar=True
        )


    def setup(self, stage: str) -> None:
        """
        Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.jit.trace(
                self.net, 
                ModelBatch(coords=Coords(x=torch.randn((16, 1))), time=torch.zeros((16, 1)))
            )
            self.net = torch.compile(self.net)


    def configure_optimizers(self) -> Dict:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        requires_grad_filter = lambda param: param.requires_grad
        optimizer = self.hparams.optimizer(params=filter(requires_grad_filter, self.parameters()))

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.conditional_loss,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        
        return {"optimizer": optimizer}