import os
import torch
from pathlib import Path
from typing import Union,NamedTuple,Dict,Any,Optional,Callable
from torch.utils.data import DataLoader,Dataset
import pickle
import sys
sys.path.append(os.getcwd())
from CMRxRecon.utils.preprocess_utils import CMRxReconPreprocessing
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
import pyas
from CMRxRecon.data.transforms import UnetDataTransform

class CMRxReconCineRawDataSample(NamedTuple):
    fname: Path
    target_path: Path
    mask_path: Path
    slice_ind: int
    dyn_ind: int
    metadata: Dict[str, Any]

class CMRxReconDataset(Dataset):
    """
    Customer Dataset for CMRxRecon challenge 2023.
    """
    def __init__(self,
                 root: Union[str, Path, os.PathLike],
                 challenge: str = 'SingleCoil',
                 task: str = 'Cine',
                 subtask: str = 'all',
                 mode: str = 'train',
                 acceleration: int = 4,
                 transform: Optional[Callable] = None,
                 ):
        if challenge not in ("SingleCoil", "MultiCoil"):
            raise ValueError('challenge should be either "SingleCoil" or "MultiCoil"')
        if task not in ("Cine", "Mapping"):
            raise ValueError('task should be either "Cine" or "Mapping"')
        elif task == 'Cine' and subtask not in ('sax','lax','all'):
            raise ValueError('subtask should be either sax, lax or all')
        elif task == 'Mapping' and subtask not in ('T1map', 'T2map', 'all'):
            raise ValueError('subtask should be either T1, T2 or all')
        if str.zfill(str(acceleration),2) not in ('04','08','10'):
            raise ValueError('acceleration should be 4, 8  or 10"')
        if mode not in ('train','validation','test'):
            raise ValueError('mode should be train, validation or test')
        if mode == 'train':
            mode_str = 'TrainingSet'
        elif mode == 'validation':
            mode_str = 'ValidationSet'
        else:
            raise NotImplementedError
        if task == 'Cine':
            if subtask == 'all':
                subtask_list = ['sax','lax']
            else:
                subtask_list = [subtask]
            subtask_list = ['_'.join([task.lower(),subtask_item]) for subtask_item in subtask_list]
        elif task == 'Mapping':
            if subtask == 'all':
                subtask_list = ['T1map','T2map']
            else:
                subtask_list = [subtask]
        
        acc_str = 'AccFactor' + str.zfill(str(acceleration),2)
        root_np_path = os.path.join(str(root) + '_np', challenge, task, mode_str + '_' + subtask,acc_str)
        self.dataset_cache_file = Path(os.path.join(os.path.dirname(root),'ChallengeDataCache','dataset_cache.pkl'))
        if not Path(root_np_path).exists() or not self.dataset_cache_file.exists():
            print(f'{root_np_path} not find. CMRxReconProcessing will be excuted!')
            CMRxReconPreprocessing(
                    root = root,
                    challenge = challenge,
                    task = task,
                    subtask = subtask,
                    mode = mode,
                    acceleration = acceleration,
                    dataset_cache_file = self.dataset_cache_file
                ).convetMat2NpwithCache()
        else:
            with open(self.dataset_cache_file,'rb') as cache_f:
                dataset_cache = pickle.load(cache_f)
            if dataset_cache.get(root_np_path) is None:
                CMRxReconPreprocessing(
                    root = root,
                    challenge = challenge,
                    task = task,
                    subtask = subtask,
                    mode = mode,
                    acceleration = acceleration,
                    dataset_cache_file = self.dataset_cache_file
                ).convetMat2NpwithCache()
        
        self.transform = transform
        self.mode = mode

        with open(self.dataset_cache_file,'rb') as cache_f:
            dataset_cache = pickle.load(cache_f)
        logging.info(f'Using dataset cache from {self.dataset_cache_file}')
        self.raw_samples = dataset_cache.get(root_np_path)

    def __len__(self):
        return len(self.raw_samples)
    
    def __getitem__(self, index):
        sample = None
        if self.mode == 'train':
            fname_np,fname_np_mask,fname_np_target = self.raw_samples[index]
            kspace = np.load(fname_np)
            target = np.load(fname_np_target)
            mask = np.load(fname_np_mask)
        elif self.mode == 'test':
            fname_np,fname_np_mask = self.raw_samples[index]
            kspace = np.load(fname_np)
            mask = np.load(fname_np_mask)
            target = None
        if self.transform is None:
            sample = (kspace,mask,target)
        else:
            sample = self.transform(kspace,mask,target)
        return sample

    
if __name__ == '__main__':
    root = 'X:\CMRxRecon\MICCAIChallenge2023\ChallengeData'
    challenge = 'SingleCoil' #MultiCoil
    myTransform = UnetDataTransform(which_challenge=challenge)
    mydata = CMRxReconDataset(root,challenge=challenge,transform=myTransform)
    mydataloader = DataLoader(mydata,batch_size = 1, shuffle = True)
    sample,mask,target= next(iter(mydataloader))
    res = np.stack([sample,target],axis = 2)
    pyas.PyArrShow().show(res)


