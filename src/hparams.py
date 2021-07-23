r"""
hparams.py

Hyper Parameters for the experiment
"""
import torch
from argparse import Namespace


def create_hparams():
    r"""
    Model hyperparamters

    Returns:
        (argparse.Namespace): Hyperparameters
    """

    data_property = torch.load("data_mean_and_std.pt")
    hparams = Namespace(

        ################################
        # Experiment Parameters        #
        ################################
        run_name="Run1",
        seed=1234,
        # Important placeholder vital to load and save model
        logger=None,
        checkpoint_path="checkpoints/",
        val_after_n_steps=100,
        # Can also have string "1,2,3" or list [1,2,3]
        gpus=[2],
        run_tests=True,
        normalize=False,


        ################################
        # Data Parameters             #
        ################################
        num_workers=0,
        data_mean=data_property["mean"].item(),
        data_std=data_property["std"].item(),
        batch_size=256,
        dataset_location="data/",
        data_max_len=63,


        ################################
        # Audio Parameters             #
        ################################
        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        ################################
        # Optimization Hyperparameters #
        ################################
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=4.0,


        ################################
        # Model Parameters             #
        ################################
        # embedding_dim = 300 etc..


    )

    return hparams
