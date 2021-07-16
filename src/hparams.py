r"""
hparams.py

Hyper Parameters for the experiment
"""
from argparse import Namespace

def create_hparams():
    r"""
    Model hyperparamters

    Returns:
        (argparse.Namespace): Hyperparameters
    """
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
        gpus=-1,
        run_tests=True,
        
        
        ################################
        # Data Parameters             #
        ################################
        num_workders=0,
        batch_size=32,
        
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