r"""
training_model.py 

This file contains PyTorch Lightning's main module where code of the main model is implemented
"""
from argparse import Namespace
from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.MelVAE import MelVAE
from src.utilities.data_utils import Normalize
from src.utilities.plotting_utils import (plot_spectrogram_to_numpy,
                                          plot_spectrogram_to_numpy_fixed_cbar)


class MyTrainingModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)
        hparams.logger = self.logger

        self.normalizer = Normalize(
            hparams.data_mean, hparams.data_std, hparams.data_max, hparams.data_min)

        self.model = MelVAE(hparams.n_channels,
                            hparams.n_kernels,
                            hparams.latent_size,
                            hparams.kernel_size,
                            hparams.stride,
                            hparams.padding,
                            hparams.img_shape
                            )

        self.recloss = nn.MSELoss()

    def kldivloss(self, mu, logsigma_sq):
        return ((mu ** 2 + logsigma_sq.exp() - 1 - logsigma_sq) / 2).mean()

    def forward(self, mel):
        r"""
        Forward pass of the model

        Args:
            x (Any): input to the forward function

        Returns:
            output (Any): output of the forward function
        """
        (mu, logsigma_sq), mel_recons = self.model(mel)

        kldivloss = self.kldivloss(mu, logsigma_sq)
        recloss = self.recloss(mel_recons, mel)

        loss = kldivloss + recloss

        return loss

    def configure_optimizers(self):
        r"""
        Configure optimizer

        Returns:
            (torch.optim.Optimizer) 
        """
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate,
                                weight_decay=self.hparams.weight_decay)

    def training_step(self, train_batch, batch_idx):
        r"""
        Main training loop of your model

        Args:
            train_batch (List): batch of input data
            batch_idx (Int): index of the current batch

        Returns:
            loss (torch.Tensor): loss of the forward run of your model
        """
        mel, mel_len = train_batch

        loss = self(mel)

        self.log("train_loss", loss.item(), prog_bar=True,
                 on_step=True, sync_dist=True, logger=True)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        r"""
        Called when the train batch ends


        Args:
            outputs (Any): output depends what you are returning from the train loop
            batch (List): batch of input data sent to the training_loop
            batch_idx (Int): index of the current batch
            dataloader_idx (Int): dataloader index
        """
        pass

    def validation_step(self, val_batch, batch_idx):
        r"""
        Validation step

        Args:
            val_batch (Any): output depends what you are returning from the train loop
            batch_idx (): batch index
        """
        mel, mel_len = val_batch

        loss = self(mel)

        self.log("val_loss", loss.item(), prog_bar=True,
                 sync_dist=True, logger=True)

        return loss

    def validation_epoch_end(self, outputs):
        r"""This is called once the validation epoch is finished

        Args:
            outputs (List[Any]): list of return from the validation step
        """

        if self.trainer.is_global_zero:
            batch = next(iter(self.val_dataloader()))
            for i in range(5):
                image = batch[0][i].to(self.device).unsqueeze(0)

                (mu, logsigma_sq), mel_recon = self.model(image)

                self.logger.experiment.add_image(
                    "InputNorm/{}".format(i+1), plot_spectrogram_to_numpy_fixed_cbar(image.squeeze().T.cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.add_image(
                    "OutputNorm/{}".format(i+1), plot_spectrogram_to_numpy_fixed_cbar(mel_recon.squeeze().T.cpu()), self.current_epoch, dataformats='HWC')

                self.logger.experiment.add_image(
                    "Output/{}".format(i+1), plot_spectrogram_to_numpy(self.normalizer.inverse_normalize(mel_recon).squeeze().T.cpu()), self.current_epoch, dataformats='HWC')
