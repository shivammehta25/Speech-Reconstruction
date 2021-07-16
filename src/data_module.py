r"""
data_module.py

Contains PyTorch-Lightning's datamodule and dataloaders 
"""
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.utilities.data_utils import CustomCollate, CustomDataset


class MyDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        r"""

        Args:
            hparams (argparse.Namespace)
        """
        super().__init__()
        self.hparams = hparams
        self.collate_fn = CustomCollate()

    def prepare_data(self):
        r"""
        Data preparation / Download Dataset
        """
        # Example
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        r"""
        Set train, val and test dataset here
        
        Args:
            stage (string, optional): fit, test based on plt.Trainer state. 
                                    Defaults to None.
        """
        # Example:
        # # Assign train/val datasets for use in dataloaders
        # if stage == 'fit' or stage is None:
        #     mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
        #     self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        #     # Optionally...
        #     # self.dims = tuple(self.mnist_train[0][0].shape)

        # # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

        #     # Optionally...
        #     # self.dims = tuple(self.mnist_test[0][0].shape)
        
        raise NotImplementedError("Add your dataloaders first and remove this line")
        
        self.train_data = CustomDataset()
        self.val_data = CustomDataset()
        self.test_data = CustomDataset()

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