import torch
from torch import nn

from typing import Dict, Callable

import torch
from torch import nn

from torchmetrics import MinMetric, MeanMetric
from torchmetrics.regression import RelativeSquaredError

from src.data.components.collate import ModelInput, ModelBatch, ModelOutput, Coords

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
            net: nn.Module,

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

        self.criterion = nn.MSELoss()

        if len(self.other_pdec) > 0:
            self.other_pdec_criterion = nn.ModuleList(
                [
                    nn.MSELoss() for _ in range(len(self.other_pdec))
                ]
            )
        else:
            self.other_pdec_criterion = None

        
        if len(self.ic) > 0:
            self.ic_criterion = nn.ModuleList(
                [
                    nn.MSELoss() for _ in range(len(self.ic))
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


    def forward(self, inputs: ModelInput) -> ModelOutput:
        return self.net(inputs)
    

    def pde(self, u, u_t, grads_1, grads_2) -> torch.Tensor:
        # Burger's equation
        return self.hparams.alpha * u_t + self.hparams.beta * u * grads_1 - self.hparams.nu * grads_2
    

    def other_pde(self, grads_1) -> torch.Tensor:
        # Continuity equation
        return grads_1


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


    def pde_forward(self, inputs: ModelInput): # PDE Loss
        u = self.forward(inputs)

        solution = u.logits

        if solution.requires_grad:
            _coords = inputs.coords.__dict__.items()

            grads_1, grads_2 = dict(), dict()

            for key, coord in _coords:
                if coord is not None:
                    # compute derivatives (gradient components)
                    grads_1[key], grads_2[key] = self.derivative(solution, coord, 2)

            # stack coordinates gradients + aggregate
            grads_1 = torch.concatenate(list(grads_1.values()), dim=-1).sum(dim=-1, keepdim=True)
            grads_2 = torch.concatenate(list(grads_2.values()), dim=-1).sum(dim=-1, keepdim=True)

            u_t = self.derivative(solution, inputs.time, 1)[0]

            pde = self.pde(solution, u_t, grads_1, grads_2)

            loss = self.criterion(pde, self.pdec(inputs))

            if self.other_pdec is not None:
                for criterion, condition in zip(self.other_pdec_criterion, self.other_pdec):

                    other_pde = self.other_pde(grads_1)

                    loss += criterion(other_pde, condition(inputs))

            return u, pde, loss
        else:
            return u, torch.tensor(0.), torch.tensor(0.)

        
    def ic_forward(self, inputs: ModelInput): # IC Loss
        ic_multi_u = list()
        ic_multi_loss = 0

        for criterion, condition in zip(self.ic_criterion, self.ic):

            ic_inputs = ModelInput(
                coords=inputs.coords,
                time=torch.zeros(size=(inputs.time.size()))
            )

            ic_multi_u.append(self.forward(ic_inputs))
            ic_multi_loss += criterion(ic_multi_u[-1].logits, condition(inputs.coords))

        return ic_multi_u, ic_multi_loss
    
    
    def bc_forward(self, inputs: ModelInput): # BC Loss
        _coords = inputs.coords.__dict__.items()

        bc_multi_u = list()
        bc_multi_loss = list()

        for index, (branch_criterion, condition) in enumerate(zip(self.bc_criterion, self.bc)):
            bc_u = list()
            bc_loss = 0

            for (key, coord), criterion in zip(_coords, branch_criterion):
                if coord is not None:
                    bc_inputs = ModelInput(
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

                    bc_loss += criterion(bc_u[-1].logits, condition(bc_inputs))

            bc_multi_u.append(bc_u)
            bc_multi_loss.append(bc_loss)

        return bc_multi_u, bc_multi_loss
    

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.monitor_loss.reset()
    
        
    def model_step(self, batch: ModelBatch) -> Tuple[torch.Tensor]:
        inputs = ModelInput(
            coords=batch.coords,
            time=batch.time
        )

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


    def training_step(self, batch: ModelBatch, batch_idx: int) -> Dict:
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
                on_step=True, on_epoch=True, prog_bar=False, sync_dist=False
            )

        return {"loss": loss}


    def validation_step(self, batch: ModelBatch, batch_idx: int) -> Dict:
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