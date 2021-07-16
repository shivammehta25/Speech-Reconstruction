r"""
data_utils.py

Utilities for processing of Data
"""
from typing import Any
import torch
from torch.utils.data.dataset import Dataset

class CustomCollate:
    r"""
    Override the __call__ method of this collate function
    """

    def __init__(self):
        super(CustomCollate, self).__init__()
        
    
    def __call__(self, batch):
        r"""
        Make changes to the batch of input, useful for tokenizing/padding on the fly

        Args:
            batch (torch.Tensor): a batch of batch_len will come here from torch.util.Dataset
        """
        raise NotImplementedError("Collate function is not implemented")
        
        x = batch[0]
        y = batch[1]
        return x, y
    
    
    
class CustomDataset(Dataset):
    def __init__(self, hparams, dataset):
        """Dataset class for torch

        Args:
            hparams (argparse.Namespace): hyperparmeters if needed for the dataset
            dataset (Any): dataset to work on
        """
        self.hparams = hparams
        self.dataset = dataset
    
    def __getitem__(self, index):
        """Return one item of the dataset

        Args:
            index (int): dataloader will fetch it batchwise, just write logic how to get one element

        Returns:
            Any: one element from the dataset
        """
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)
