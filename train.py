r"""
train.py 

PyTorch-Lightning Trainer file, main file to run your experiments with
"""
import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.hparams import create_hparams
from src.data_module import MyDataModule
from src.training_model import MyTrainingModule
from run_tests import run_tests

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    args = parser.parse_args()

    if args.checkpoint_path and not os.path.exists(args.checkpoint_path):
        raise FileExistsError("Check point not present recheck the name")

    hparams = create_hparams()

    if hparams.run_tests:
        run_tests()

    
    seed_everything(hparams.seed)

    data_module = MyDataModule(hparams)
    model = MyTrainingModule(hparams)
    
    
    
    
    ## TODO: Uncomment to only load the model parameters
    # model = Tacotron2Trainer.load_from_checkpoint("checkpoints/DataDropout95-epoch=9-step=7018-ver=0.ckpt")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=hparams.checkpoint_path,
                                          filename='{}-{}-{}-{}-{:.2f}'.format(hparams.run_name, epoch, step, ver, val_loss),
                                          verbose=True,
                                          every_n_val_epochs=1)

    logger = TensorBoardLogger('tb_logs', name=hparams.run_name)

    trainer = pl.Trainer(resume_from_checkpoint=args.checkpoint_path,
                         default_root_dir=os.path.join("checkpoints", hparams.run_name),
                         gpus=hparams.gpus,
                         logger=logger,
                         log_every_n_steps=1,
                         flush_logs_every_n_steps=10,
                         plugins=DDPPlugin(find_unused_parameters=False),
                         accelerator='ddp',
                         val_check_interval=hparams.val_after_n_steps,
                         gradient_clip_val=hparams.grad_clip_thresh,
                         callbacks=[checkpoint_callback],
                         track_grad_norm=2,
                         )

    trainer.fit(model, data_module)
