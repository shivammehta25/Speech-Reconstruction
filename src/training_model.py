r"""
training_model.py

This file contains PyTorch Lightning's main module where code of the main model is implemented
"""
from typing import List, Tuple, Any
from src.utilities.plotting_utils import (plot_spectrogram_to_numpy,
                                          plot_spectrogram_to_numpy_fixed_cbar)
from src.utilities.data_utils import Normalize
from src.model.MelVAE import MelVAE
from src.model.MelGAN import Generator, Discriminator
from src.model.MelDCGAN import Generator_Conv, Discriminator_Conv
import torch.nn.functional as F
import torch.nn as nn
import torch
import pytorch_lightning as pl
from typing import Any, List, Tuple
from argparse import Namespace
import torchvision
from collections import OrderedDict


class MelVAETrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(MelVAETrainer, self).__init__()
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


class MelGANTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(MelGANTrainer, self).__init__()

        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)
        hparams.logger = self.logger

        self.normalizer = Normalize(
            hparams.data_mean, hparams.data_std, hparams.data_max, hparams.data_min)
        self.save_hyperparameters()

        # networks
        data_shape = (hparams.n_channels, *hparams.img_shape)
        self.generator = Generator(
            latent_dim=self.hparams.latent_size, img_shape=data_shape)
        self.discriminator = Discriminator(img_shape=data_shape)

        self.register_buffer("validation_z", torch.randn(
            5, self.hparams.latent_size))

        self.example_input_array = torch.zeros(2, self.hparams.latent_size)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_size)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict(
                {'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            self.log("loss/g_loss", g_loss.item())
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict(
                {'loss': d_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            self.log("loss/d_loss", d_loss.item())
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr_gan
        b1 = self.hparams.b1_gan
        b2 = self.hparams.b2_gan

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        with torch.no_grad():
            z = self.validation_z
            # log sampled images
            sample_imgs = self(z)
            for i, mel_recon in enumerate(sample_imgs):
                self.logger.experiment.add_image(
                    "OutputNorm/{}".format(i+1), plot_spectrogram_to_numpy_fixed_cbar(mel_recon.squeeze().T.cpu()), self.current_epoch, dataformats='HWC')

                self.logger.experiment.add_image(
                    "Output/{}".format(i+1), plot_spectrogram_to_numpy(self.normalizer.inverse_normalize(mel_recon).squeeze().T.cpu()), self.current_epoch, dataformats='HWC')


class MelDCGANTrainer(pl.LightningModule):

    def __init__(self, hparams):
        super(MelDCGANTrainer, self).__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)
        hparams.logger = self.logger

        self.normalizer = Normalize(
            hparams.data_mean, hparams.data_std, hparams.data_max, hparams.data_min)
        self.save_hyperparameters()

        # networks
        data_shape = (hparams.n_channels, *hparams.img_shape)
        self.generator = Generator_Conv(
            latent_dim=self.hparams.latent_size, img_shape=data_shape)

        self.discriminator = Discriminator_Conv(img_shape=data_shape)

        self.register_buffer("validation_z", torch.randn(
            5, self.hparams.latent_size))
        self.example_input_array = torch.zeros(2, self.hparams.latent_size)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, _ = batch

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_size)
        z = z.type_as(imgs)

        # train generator
        if optimizer_idx == 0:

            # generate images
            self.generated_imgs = self(z)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log("loss/g_loss", g_loss.item())

            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(
                self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            self.log("loss/d_loss", d_loss.item())

            return output

    def configure_optimizers(self):
        lr = self.hparams.lr_gan
        b1 = self.hparams.b1_gan
        b2 = self.hparams.b2_gan

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        with torch.no_grad():
            z = self.validation_z
            # log sampled images
            sample_imgs = self(z)
            for i, mel_recon in enumerate(sample_imgs):
                self.logger.experiment.add_image(
                    "OutputNorm/{}".format(i+1), plot_spectrogram_to_numpy_fixed_cbar(mel_recon.squeeze().T.cpu()), self.current_epoch, dataformats='HWC')

                self.logger.experiment.add_image(
                    "Output/{}".format(i+1), plot_spectrogram_to_numpy(self.normalizer.inverse_normalize(mel_recon).squeeze().T.cpu()), self.current_epoch, dataformats='HWC')
