r"""
data_module.py

Contains PyTorch-Lightning's datamodule and dataloaders 
"""
import os

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from src.utilities.data_utils import CustomCollate, Normalize, SpeechCommandDataset


class MyDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        r"""

        Args:
            hparams (argparse.Namespace)
        """
        super().__init__()
        self.hparams = hparams
        self.collate_fn = CustomCollate(hparams.data_max_len)

    def prepare_data(self):
        r"""
        Data preparation / Download Dataset
        """
        # Example
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)
        SPEECHCOMMANDS(self.hparams.dataset_location, download=True)

    def load_list(self, filename):
        filepath = os.path.join(self._path, filename)
        with open(filepath) as fileobj:
            return [os.path.join(self._path, line.strip()) for line in fileobj]

    def setup(self, stage=None):
        r"""
        Set train, val and test dataset here

        Args:
            stage (string, optional): fit, test based on plt.Trainer state. 
                                    Defaults to None.
        """

        self.train_data = SpeechCommandDataset(self.hparams, subset="training")
        self.val_data = SpeechCommandDataset(self.hparams, subset="validation")
        self.test_data = SpeechCommandDataset(self.hparams, subset="testing")

    def train_dataloader(self):
        r"""
        Load trainset dataloader

        Returns:
            (torch.utils.data.DataLoader): Train Dataloader
        """
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        r"""
        Load Validation dataloader

        Returns:
            (torch.utils.data.DataLoader): Validation Dataloader
        """

        return DataLoader(self.val_data, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        r"""
        Load Test dataloader

        Returns:
            (torch.utils.data.DataLoader): Test Dataloader
        """
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, collate_fn=self.collate_fn,
                          num_workers=self.hparams.num_workers, pin_memory=True)
