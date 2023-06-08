import pytorch_lightning as pl
from pathlib import Path
from argparse import ArgumentParser
from typing import Callable
from torch.utils.data import DataLoader, DistributedSampler
from CMRxRecon.data.mri_data import CMRxReconDataset

from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context #for window multi-worker only


class CMRxReconDataModule(pl.LightningDataModule):
    """
    Data module class for CMRxRecon challenge 2023.

    This class handles configuration for traning on CMRxRecon data.
    """
    def __init__(
            self,
            data_path: Path,
            train_transform: Callable,
            val_transform: Callable,
            challenge: str = 'SingleCoil',
            task: str = 'Cine',
            sub_task: str = 'all',
            acceleration: int = 4,
            use_dataset_cache_file: bool = True, #not recommaned to use option: False
            batch_size: int = 1,
            num_workers: int = 4,
            distributed_sampler: bool = False,
    ):
        """
        Args:
            data_path: Path to root data directory.
            train_transform: Callable; A tranform object for the training data.
            val_transform: Callable; A transform object for the validation data. 
            challenge: Name of challenge from ('SingleCoil','MultiCoil').
            task: Name of task from ('Cine','Mapping').
            sub_task: Name of sub_task for 'Cine' from ('lax','sax','all')
            acceleration: Acceleration factor for sub_task, e.g. 4,8,10 for 'Cine'
            use_dataset_cache_file: bool; if dataset_cache is used.
            batch_size: Batch size.
            num_workers: Number of workers in PyTorch dataloader.
            distributed_sampler: Whether to use a distributed sampler. This should be set
                True if training with ddp 
        """
        super().__init__()

        self.data_path = data_path
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.challenge = challenge
        self.task = task
        self.sub_task = sub_task
        self.acceleration = acceleration
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
    
    def _create_data_loader(
            self,
            data_transform: Callable,
            data_partition: str)->DataLoader:
        dataset = CMRxReconDataset(
            root = self.data_path,
            challenge = self.challenge,
            task = self.task,
            subtask = self.sub_task,
            mode = data_partition,
            acceleration = self.acceleration,
            transform = data_transform)
        
        sampler = None
        if self.distributed_sampler:
            if data_partition == 'train':
                sampler = DistributedSampler(dataset)
        
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                sampler=sampler,
                                shuffle = True if data_partition == 'train' else False,
                                multiprocessing_context= get_context('loky'),
                                pin_memory=True)
        return dataloader

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._create_data_loader(self.train_transform,data_partition='train')
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._create_data_loader(self.val_transform,data_partition='validation')
    
    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return self._create_data_loader(self.)
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Define parameters that apply to the data
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        #dataset argumenets
        parser.add_argument(
            '--data_path',
            default = None,
            type = Path,
            help = "Path to CMRxRecon data root"
        )
        parser.add_argument(
            '--challenge',
            choices = ('SingleCoil','MultiCoil'),
            default= 'SingleCoil',
            type = str,
            help = 'Wich challeneg to preprocess for.'
        )
        parser.add_argument(
            '--task',
            choices=('Cine','Mapping'),
            default='Cine',
            type = str,
            help = 'Which task to preprocess for.'
        )
        parser.add_argument(
            '--sub_task',
            choices=('sax','lax','all','T1map','T2map'),
            default= 'all',
            type = str,
            help = "Which sub_task to preprocess for."
        )
        parser.add_argument(
            '--acceleration',
            choices=(4,8,10),
            default=4,
            type=int,
            help="Which acceleration for subtask."
        )
        parser.add_argument(
            '--use_dataset_cache_file',
            default=True,
            type = bool,
            help = "If init dataset by using dataset cache."
            )
        parser.add_argument(
            '--batch_size',
            default=1,
            type=int,
            help="Data loader batch size"
        )
        parser.add_argument(
            '--num_workers',
            default = 4,
            type=int,
            help='Number of workers to use in data loader'
        )

        return parser