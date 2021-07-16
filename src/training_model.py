r"""
training_model.py 

This file contains PyTorch Lightning's main module where code of the main model is implemented
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import List, Tuple, Any
from argparse import Namespace

from src.model.MainModel import MainModel

class MyTrainingModule(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        if type(hparams) != Namespace:
            hparams = Namespace(**hparams)

        self.save_hyperparameters(hparams)        
        hparams.logger = self.logger
        
        # TODO: Change this to your pytorch model
        self.model = MainModel(hparams)

        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, x):
        r"""
        Forward pass of the model

        Args:
            x (Any): input to the forward function

        Returns:
            output (Any): output of the forward function
        """
        output = self.model(x)
        return output
    
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
        x, y = train_batch
        output = self(x)
        loss = self.loss(output, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, sync_dist=True, logger=True)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        r"""
        Called when the train batch ends
        
        TODO: If you want to do something after your batch iteration
        
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
        x, y = val_batch
        output = self(x)
        loss = self.loss(output, y)
        self.log("val_loss", loss, prog_bar=True, on_step=True, sync_dist=True, logger=True)
        return loss
    
    def validation_epoch_end(self, outputs):
        r"""This is called once the validation epoch is finished

        Args:
            outputs (List[Any]): list of return from the validation step
        """
        
        # TODO: Plots etc. are saved here
        pass