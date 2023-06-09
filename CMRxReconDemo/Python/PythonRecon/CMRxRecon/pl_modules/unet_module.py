from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .mri_module import MRIModule
import torch
from torch.nn import functional as F
from argparse import ArgumentParser
from CMRxRecon.models.unet import Unet
from CMRxRecon import evaluate

class UnetModule(MRIModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """
    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            chans: int = 32,
            num_pool_layers: int = 4,
            drop_prob: float = 0.0,
            lr=0.001,
            lr_step_size=40,
            lr_gamma=0.1,
            weight_decay=0.0,
            **kwargs):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.unet = Unet(in_chans=self.in_chans,
                         out_chans=self.out_chans,
                         chans=self.chans,
                         num_pool_layers=self.num_pool_layers,
                         drop_prob=self.drop_prob)
    def forward(self,image):
        return self.unet(image.unsqueeze(1)).squeeze(1)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x,y = batch.image, batch.target
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log("loss", loss.detach())
        return loss

    def validation_step(self,batch,batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx":batch_idx,
            "fname":batch.fname,
            "input":batch.image * std + mean,
            "output": output*std + mean,
            "target": batch.target * std + mean,
            "val_loss": F.l1_loss(output,batch.target)
        }

    
    def configure_optimizers(self) -> Any:
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Defines parameters only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser],add_help=False)
        parser = MRIModule.add_model_specific_args(parser)

        #network params
        parser.add_argument('--in_chans',
                            default = 1,
                            type = int,
                            help = "Number of UNet input channels.")
        parser.add_argument('--out_chans',
                            default = 1,
                            type = int,
                            help = "Number of UNet output channels.")
        parser.add_argument('--chans',
                            default = 32,
                            type = int,
                            help = "Number of top-level U-Net filters.")
        parser.add_argument('--num_pool_layers',
                            default = 4,
                            type = int,
                            help = "Number of U-Net pooling layers.")
        parser.add_argument('--drop_prob',
                            default = 0.0,
                            type = float,
                            help = "U-Net dropout probability")
        parser.add_argument('--lr',
                            default = 0.001,
                            type = float,
                            help = "RMSProp learning rate")
        parser.add_argument('--lr_step_size',
                            default = 40,
                            type = float,
                            help = "Epoch at which to decrease step size")
        parser.add_argument('--lr_gamma',
                            default = 0.1,
                            type = float,
                            help = "Amount to decrease step size")
        parser.add_argument('--weight_decay',
                            default = 0.0,
                            type = float,
                            help = "Strength of weight decay regularization")
        return parser