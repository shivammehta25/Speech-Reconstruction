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
<<<<<<< HEAD
        run_name="NoSigmoid",
=======
        run_name="Run1",
>>>>>>> cc5b82bb4a95c6908ceb29e00ad530655be06c6e
        seed=1234,
        # Important placeholder vital to load and save model
        logger=None,
        checkpoint_path="checkpoints/",
<<<<<<< HEAD
        val_every_n_epoch=1,
=======
        val_after_n_steps=100,
>>>>>>> cc5b82bb4a95c6908ceb29e00ad530655be06c6e
        # Can also have string "1,2,3" or list [1,2,3]
        gpus=[2],
        run_tests=True,
        normalize=False,


        ################################
        # Data Parameters             #
        ################################
<<<<<<< HEAD
        num_workers=32,
        data_mean=data_property["mean"].item(),
        data_std=data_property["std"].item(),
        data_max=data_property["max"].item(),
        data_min=data_property["min"].item(),
        batch_size=1024,
        dataset_location="data/",
        data_max_len=80,
=======
        num_workers=0,
        data_mean=data_property["mean"].item(),
        data_std=data_property["std"].item(),
        batch_size=256,
        dataset_location="data/",
        data_max_len=63,
>>>>>>> cc5b82bb4a95c6908ceb29e00ad530655be06c6e


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
<<<<<<< HEAD
        n_channels=1,
        n_kernels=512,
        latent_size=256,
        kernel_size=4,
        stride=2,
        padding=1,
        img_shape=(80, 80),
=======
        # embedding_dim = 300 etc..
>>>>>>> cc5b82bb4a95c6908ceb29e00ad530655be06c6e


    )

    return hparams
