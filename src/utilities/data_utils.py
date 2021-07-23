r"""
data_utils.py

Utilities for processing of Data
"""
from src.utilities.audio.audio_processing import TacotronSTFT
from torchaudio.datasets.speechcommands import SPEECHCOMMANDS
from torch.nn.functional import normalize
import os
from argparse import Namespace
from typing import Any, Union

import torch
import torch.nn as nn


def pad_sequence(batch, max_length):
    # Make all tensor in a batch the same length by padding with zeros
    new_batch, len_of_items = [], []
    for item in batch:
        new_batch.append(item.t())
        len_of_items.append(len(item.t()))

    new_batch.append(torch.zeros(max_length, batch[0].shape[0]))

    batch = torch.nn.utils.rnn.pad_sequence(
        new_batch, batch_first=True, padding_value=0.)
    return batch[:-1], torch.LongTensor(len_of_items)


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

        mel, mel_len = pad_sequence(batch, self.data_max_len)
        return mel.unsqueeze(1), mel_len


class Normalize(nn.Module):
    r"""
    Normalize Module
    ---

    Responsible for normalization and denormalization of inputs
    """

    def __init__(self, mean: float, std: float, max: float, min: float):
        super(Normalize, self).__init__()

        if not torch.is_tensor(mean):
            mean = torch.tensor(mean)
        if not torch.is_tensor(std):
            std = torch.tensor(std)

        if not torch.is_tensor(max):
            max = torch.tensor(max)
        if not torch.is_tensor(min):
            min = torch.tensor(min)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("max", max)
        self.register_buffer("min", min)

    def forward(self, x):
        r"""
        Takes an input and normalises it
        """
        return self._standardize(x)

    def inverse_normalize(self, x):
        r"""
        Takes an input and de-normalises it
        """
        return self._inverse_standardize(x)

    def _standardize(self, x):

        if not torch.is_tensor(x):
            x = torch.tensor(x)

        x = x.sub(self.mean).div(self.std)
        return x

    def _inverse_standardize(self, x):
        r"""
        Takes an input and de-normalises it
        """
        if not torch.is_tensor(x):
            x = torch.tensor([x])

        x = x.mul(self.std).add(self.mean)
        return x

    def _min_max_normalize(self, x):
        x = x.sub(self.min).div(self.max.sub(self.min))
        return x

    def _min_max_denormalize(self, x):
        x = x.mul(self.max.sub(self.min)).add(self.min)
        return x


class SpeechCommandDataset(SPEECHCOMMANDS):

    def __init__(self, hparams, subset=None, normalize=True) -> None:

        super(SpeechCommandDataset, self).__init__(
            root=hparams.dataset_location, subset=subset)

        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)

        self.normalize = normalize

        self.normalizer = Normalize(
            hparams.data_mean, hparams.data_std, hparams.data_max, hparams.data_min)

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

        return self.normalizer(melspec) if self.normalize else melspec

    def __len__(self) -> int:
        return super().__len__()
