r"""
data_utils.py

Utilities for processing of Data
"""
import os
from argparse import Namespace
from typing import Any, Union

import torch
import torch.nn as nn
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from src.utilities.audio.audio_processing import TacotronSTFT


def pad_sequence(batch, max_length):
    # Make all tensor in a batch the same length by padding with zeros
    new_batch, len_of_items = [], []
    for item in batch:
        new_batch.append(item.t())
        len_of_items.append(len(item.t()))

    batch = torch.nn.utils.rnn.pad_sequence(
        new_batch, batch_first=True, padding_value=0.)
    return batch, torch.LongTensor(len_of_items)


class CustomCollate:
    r"""
    Override the __call__ method of this collate function
    """

    def __init__(self, data_max_len):
        super(CustomCollate, self).__init__()

        self.data_max_len = data_max_len

    def __call__(self, batch):
        r"""
        Make changes to the batch of input, useful for tokenizing/padding on the fly

        Args:
            batch (torch.Tensor): a batch of batch_len will come here from torch.util.Dataset
        """

        return pad_sequence(batch, self.data_max_len)


class Normalize(nn.Module):
    r"""
    Normalize Module
    ---

    Responsible for normalization and denormalization of inputs
    """

    def __init__(self, mean: float, std: float):
        super(Normalize, self).__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        return self._normalize(x)

    def _normalize(self, x):
        r"""
        Takes an input and normalises it
        """
        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def inverse_normalize(self, x):
        """
        Takes an input and de-normalises it
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x


class SpeechCommandDataset(SPEECHCOMMANDS):

    def __init__(self, hparams, subset=None) -> None:

        super(SpeechCommandDataset, self).__init__(
            root=hparams.dataset_location, subset=subset)

        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.normalizer = Normalize(hparams.data_mean, hparams.data_std)

    def __getitem__(self, n: int):
        data_item = super().__getitem__(n)
        audio, sampling_rate, *_ = data_item

        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))

        # audio_norm = audio / self.max_wav_value
        # audio_norm = audio_norm.unsqueeze(0)

        melspec = self.stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)

        return self.normalizer(melspec)

    def __len__(self) -> int:
        return super().__len__()
