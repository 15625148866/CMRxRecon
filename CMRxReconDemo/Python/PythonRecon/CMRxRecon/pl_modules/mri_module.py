"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import pathlib
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from torchmetrics.metric import Metric

from CMRxRecon import evaluate


class DistributedMetricSum(Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("quantity", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, batch: torch.Tensor):  # type: ignore
        self.quantity += batch

    def compute(self):
        return self.quantity
    
class MRIModule(pl.LightningModule):
    """
    Abstract super class for deep learning reconstruction models.

    This is a subclass of the LightningModule class from pytorch_lightning,
    with some additional functionality specific to CMRxRecon Challenge 2023:
        - Evaluating reconstructions
        - Visualization

    To implement a new reconstruction model, inherit from this class and
    implement the following methods:
        - training_step, validation_step, test_step:
            Define what happens in one step of training, validation, and
            testing, respectively
        - configure_optimizers:
            Create and return the optimizers
    
    Other methods from LightningModule can be overridden as needed.
    """
    def __init__(self,num_log_images: int = 16):
        """
        Args:
            num_log_images: Number of images to log. Defaults to 16.
        """
        super().__init__()
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.NMSE = DistributedMetricSum()
        self.SSIM = DistributedMetricSum()
        self.PSNR = DistributedMetricSum()
        self.ValLoss = DistributedMetricSum()
        self.TotExamples = DistributedMetricSum()
        self.TotSliceExamples = DistributedMetricSum()
    
    def validation_step_end(self, val_logs):

        #pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            self.val_log_indices = list(
                np.random.permutation(len(self.trainer.val_dataloaders[0]))[
                    :self.num_log_images
                ]
            )
        
        #log images to tensorboard
        if isinstance(val_logs['batch_idx'],int):
            batch_indices = [val_logs['batch_idx']]
        else:
            batch_indices = val_logs['batch_idx']
        for i,batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                key = f"val_images_id_{batch_idx}"
                target = val_logs['target'][i].unsqueeze(0)
                output = val_logs['output'][i].unsqueeze(0)
                error = torch.abs(target - output)
                output = output / output.max()
                target = target / target.max()
                error = error / error.max()
                self.log_image(f'{key}/target',target)
                self.log_image(f'{key}/reconstruction',output)
                self.log_image(f'{key}/error',error)
        
        #compute evaluation metrics
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()
        for i, fname in enumerate(val_logs['fname']):
            slice_num = int(val_logs['slice_num'][i].cpu())
            maxval = val_logs['max_value'][i].cpu().numpy()
            output = val_logs['output'][i].cpu().numpy()
            target = val_logs['target'][i].cpu().numpy()

            mse_vals[fname][slice_num] = torch.tensor(
                evaluate.mse(target,output)
            ).view(1)
            target_norms[fname][slice_num] = torch.tensor(
                evaluate.mse(target, np.zeros_like(target))
            ).view(1)
            ssim_vals[fname][slice_num] = torch.tensor(
                evaluate.ssim(target[None,...],output[None,...], maxval = maxval)
            ).view(1)
            max_vals[fname] = maxval
        
        return {
            "val_loss": val_logs['val_loss'],
            'mse_vals': dict(mse_vals),
            'target_norms': dict(target_norms),
            'ssim_vals': dict(ssim_vals),
            'max_vals': max_vals
        }
        
        
    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, global_step = self.global_step)

    def validation_epoch_end(self, val_logs):
        losses = []
        mse_vals = defaultdict(dict)
        target_norms = defaultdict(dict)
        ssim_vals = defaultdict(dict)
        max_vals = dict()

        #use dict updates to handle duplicate slices
        for val_log in val_logs:
            losses.append(val_log["val_loss"].view(-1))

            for k in val_log["mse_vals"].keys():
                mse_vals[k].update(val_log["mse_vals"][k])
            for k in val_log["target_norms"].keys():
                target_norms[k].update(val_log["target_norms"][k])
            for k in val_log["ssim_vals"].keys():
                ssim_vals[k].update(val_log["ssim_vals"][k])
            for k in val_log["max_vals"]:
                max_vals[k] = val_log["max_vals"][k]
        
        #check to make sure we have all files in all metrics
        assert(
            mse_vals.keys()
            == target_norms.keys()
            == ssim_vals.keys()
            == max_vals.keys()
        )

        #apply means across image volumes
        metrics = {"nmse":0, "ssim":0, "psnr":0}
        local_example = 0
        for fname in mse_vals.keys():
            local_example = local_example + 1
            mse_val = torch.mean(
                torch.cat([v.view(-1) for _,v in mse_vals[fname].items()])
            )
            target_norm = torch.mean(
                torch.cat([v.view(-1) for _,v in target_norms[fname].items()])
            )
            metrics["nmse"] = metrics["nmse"] + mse_val / target_norm
            metrics['psnr'] = (
                metrics['psnr']
                + 20
                * torch.log10(
                    torch.tensor(
                        max_vals[fname],dtype=mse_val.dtype,device=mse_val.device
                    )
                )
                - 10 * torch.log10(mse_val)
            )
            metrics['ssim'] = metrics['ssim'] + torch.mean(
                torch.cat([v.view(-1) for _,v in ssim_vals[fname].items()])
            )
        
        #reduce across ddp via sum
        metrics['nmse'] = self.NMSE(metrics['nmse'])
        metrics['ssim'] = self.SSIM(metrics['ssim'])
        metrics['psnr'] = self.PSNR(metrics['psnr'])
        tot_examples = self.TotExamples(torch.tensor(local_example))
        val_loss = self.ValLoss(torch.sum(torch.cat(losses)))
        tot_slice_examples = self.TotSliceExamples(
            torch.tensor(len(losses),dtype=torch.float)
        )

        self.log('validation_loss', val_loss/ tot_slice_examples, prog_bar=True)
        for metric, value in metrics.items():
            self.log(f"val_metrics/{metric}",value/tot_examples)

    
    @staticmethod
    def add_model_specific_args(parent_args):
        """
        Define parameters that only apply to this module.
        """
        parser = ArgumentParser(parents = [parent_args], add_help = False)

        #logging params
        parser.add_argument(
            '--num_log_images',
            default = 16,
            type = int,
            help = "Number of images to log to Tensorboard"
        )

        return parser



    