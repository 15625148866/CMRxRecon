from argparse import ArgumentParser
import sys
import os
sys.path.append(os.getcwd())
from CMRxRecon.pl_modules.data_module import CMRxReconDataModule
from CMRxRecon.pl_modules.unet_module import UnetModule
from CMRxRecon.data.transforms import UnetDataTransform
from CMRxRecon.data.mri_data import CMRxReconDataset
import pytorch_lightning as pl
import time
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, SubsetRandomSampler
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context #for window multi-worker only
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np

def cli_main(args):
    train_transform = UnetDataTransform(args.challenge)
    # val_transform = UnetDataTransform(args.challenge)
    #ptl data module - this handles data loaders
    CMRxSingleSliceSingleDynDataset = CMRxReconDataset(
            root = args.data_path,
            challenge = args.challenge,
            task = args.task,
            subtask = args.sub_task,
            mode = args.mode,
            acceleration = args.acceleration,
            transform = train_transform)
    k_fold = KFold(n_splits=5,shuffle=True,random_state=42)
    
    for i,(train_idx, val_idx) in enumerate(k_fold.split(np.arange(len(CMRxSingleSliceSingleDynDataset)))):
        train_sampler = SubsetRandomSampler(train_idx[:32])
        val_sampler = SubsetRandomSampler(val_idx[:32])
        train_loader = DataLoader(CMRxSingleSliceSingleDynDataset,batch_size=args.batch_size,sampler=train_sampler)
        val_loader = DataLoader(CMRxSingleSliceSingleDynDataset, batch_size=args.batch_size, sampler=val_sampler)
        tb_logger = TensorBoardLogger(save_dir=os.path.join(os.path.dirname(args.data_path),'logs'),name = f'fold_{i+1}')
        checkpoint_callback = ModelCheckpoint(dirpath=tb_logger.log_dir,
                                              filename="{epoch:02d}--{val_metric:.4f}",
                                              monitor = "validation_loss",
                                              mode = "min")
        # ------------
        # model
        # ------------
        model = UnetModule(
            in_chans=args.in_chans,
            out_chans=args.out_chans,
            chans=args.chans,
            num_pool_layers=args.num_pool_layers,
            drop_prob=args.drop_prob,
            lr=args.lr,
            lr_step_size=args.lr_step_size,
            lr_gamma=args.lr_gamma,
            weight_decay=args.weight_decay,
        )

        # ------------
        # trainer
        # ------------
        trainer = pl.Trainer(max_epochs=args.max_epochs,
                             devices=args.devices,
                             accelerator=args.accelerator,
                             callbacks= [checkpoint_callback],
                             logger = tb_logger)
        # trainer = pl.Trainer.from_argparse_args(args)

        # ------------
        # run
        # ------------
        trainer.fit(model, train_dataloaders=train_loader,val_dataloaders=val_loader)
    # if args.mode == "train":
    #     trainer.fit(model, datamodule=data_module)
    # elif args.mode == "test":
    #     trainer.test(model, datamodule=data_module)
    # else:
    #     raise ValueError(f"unrecognized mode {args.mode}")



def build_args():
    parser = ArgumentParser()

    num_gpus = 1
    backend = 'ddp'
    accelerator = 'gpu'
    devices = 'auto'
    batch_size = 32
    root = r'X:\CMRxRecon\MICCAIChallenge2023\ChallengeData'
    default_root_dir = os.path.join(root,'logs','unet')

    parser.add_argument(
        '--mode',
        default="train",
        choices=("train","test"),
        type=str,
        help='Operation mode'
    )

    parser = CMRxReconDataModule.add_data_specific_args(parser)
    parser.set_defaults(data_path = root, batch_size = batch_size)

    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
    )

    #trainer config
    parser = pl.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        gpus=num_gpus,  # number of gpus to use
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        # strategy=backend,  # what distributed version to use
        accelerator = accelerator,  #KX
        devices = devices,  #KX
        auto_select_gpus = True,
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_root_dir,  # directory for logs and checkpoints
        max_epochs=1,  # max number of epochs
    )

    args = parser.parse_args()
    checkpoint_dir = Path(os.path.join(args.default_root_dir,"checkpoints"))
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    
    # args.callbacks = [
    #     pl.callbacks.ModelCheckpoint(
    #         dirpath = os.path.join(args.default_root_dir,"checkpoints"),
    #         save_top_k = True,
    #         verbose = True,
    #         monitor = "validation_loss",
    #         mode = "min"
    #     )
    # ]

    #set default checkpoint if one exists in our checkpoint directory
    # if args.resume_from_checkpoint is None:
    #     ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"),key=os.path.getmtime)
    #     if ckpt_list:
    #         args.resume_from_checkpoint = str(ckpt_list[-1])
    return args


def run_cli():
    args = build_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    cli_main(args)


if __name__ == '__main__':
    run_cli()