r"""
data_utils.py

Utilities for processing of Data
"""
import os
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torchaudio.datasets import SPEECHCOMMANDS


class SpeechCommandDataset(SPEECHCOMMANDS):
    r"""
    Speech Command dataset
    """

    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + \
                load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]


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


class SpeechCommands(SPEECHCOMMANDS):
    def __init__(self, hparams: Namespace, subset: str = None):
        """Dataset class for torch

        Args:
            hparams (argparse.Namespace): hyperparmeters if needed for the dataset
            subset (str): one of [training, testing, validation]
        """
        self.hparams = hparams
        self.dataset = SpeechCommandDataset(subset)

    def __getitem__(self, index):
        """Return one item of the dataset

        Args:
            index (int): dataloader will fetch it batchwise, just write logic how to get one element

        Returns:
            Any: one element from the dataset
        """
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[index]

        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class Normalize(nn.Module):
    r"""
    Normalize Module
    ---

    Responsible for normalization and denormalization of inputs
    """

    def __init__(self, mean: int, std: int):
        super(Normalize, self).__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def __call__(self, x):
        return self._normalize(x)

    def _normalize(self, x):
        r"""
        Takes an input and normalises it
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def inverse_normalise(self, x):
        """
        Takes an input and de-normalises it
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x
