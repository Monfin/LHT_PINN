from lightning import LightningDataModule
import torch

import numpy as np

import hydra

# from torch.utils.data import random_split
from src.data.components.collate import BaseCollator

from typing import Optional, Any

import logging
log = logging.getLogger(__name__)


class LitDataModule(LightningDataModule):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,

            collator: BaseCollator,

            train_batch_size: int = 128,
            val_batch_size: int = 128,

            pin_memory: bool = False,
            num_workers: int = 0,
            persistent_workers: bool = False
    ) -> None:
        super(LitDataModule, self).__init__()
        
        self.save_hyperparameters()

        self.dataset = hydra.utils.instantiate(dataset)

        self.data_train: Optional[torch.utils.data.Dataset] = None
        self.data_val: Optional[torch.utils.data.Dataset] = None


    @property
    def num_classes(self):
        return 0


    def prepare_data(self) -> None:
        """
        Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass


    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        log.info("Setup data...")

        self.train_data, self.val_data = torch.utils.data.random_split(
            self.dataset, 
            [0.8, 0.2]
        )


    def train_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """
        Create and return the train dataloader.

        :return: The train dataloader.
        """

        collator = hydra.utils.instantiate(self.hparams.collator)

        return torch.utils.data.DataLoader(
            dataset=self.train_data,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collator,
            shuffle=True,
            persistent_workers=self.hparams.persistent_workers
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[Any]:
        """
        Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        collator = hydra.utils.instantiate(self.hparams.collator)

        return torch.utils.data.DataLoader(
            dataset=self.val_data,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collator,
            shuffle=False,
            persistent_workers=self.hparams.persistent_workers
        ) 