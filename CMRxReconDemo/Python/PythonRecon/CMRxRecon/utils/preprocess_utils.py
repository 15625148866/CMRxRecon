import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from CMRxRecon.utils.math_utils import loadmat
from typing import Union,Optional
from pathlib import Path
import os
import pickle
import logging
import pandas as pd

class CMRxReconPreprocessing:
    def __init__(self,
                root: Union[str, Path, os.PathLike],
                challenge: str = 'SingleCoil',
                task: str = 'Cine',
                subtask: str = 'all',
                mode: str = 'train',
                acceleration: int = 4,
                dataset_cache_file: Union[str,Path,os.PathLike] = "dataset_cache.pkl",
                ):
        #input check
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
        
        #save to local pars
        self.root = root
        self.challenge = challenge
        self.task = task
        self.subtask = subtask
        self.subtask_list = subtask_list
        self.acceleration = acceleration
        self.mode = mode
        self.mode_str = mode_str
        self.dataset_cache_file = Path(dataset_cache_file)
        #help path
        self.root_path = os.path.join(self.root, self.challenge, self.task, self.mode_str)
        self.acc_str = 'AccFactor' + str.zfill(str(self.acceleration),2)
        self.target_str = 'FullSample'
        
    def convetMat2NpwithCache(self):
        self.root_np_path = os.path.join(str(self.root) + '_np', self.challenge, self.task, self.mode_str + '_' + self.subtask,self.acc_str)
        if not Path(self.root_np_path).exists():
            os.makedirs(self.root_np_path)

        if not Path(os.path.dirname(self.dataset_cache_file)).exists():
            os.makedirs(os.path.dirname(self.dataset_cache_file))
        self.raw_samples = []

        #remove data to path
        files = list(Path(os.path.join(self.root_path,self.acc_str)).iterdir())
        for file in sorted(files):#[files[0]]
            for subtask_item in self.subtask_list:
                #add postfix of .mat for given mat file name
                fname = Path(os.path.join(file,subtask_item + '.mat'))
                if fname.exists():
                    data = loadmat(fname)
                    data = data[list(data.keys())[0]]
                else:
                    continue
                mask_path = Path(os.path.join(file,subtask_item +'_mask' + '.mat'))
                target_path = Path(os.path.join(self.root_path,self.target_str,os.path.basename(file),subtask_item + '.mat'))
                mask = loadmat(mask_path)
                mask = mask[list(mask.keys())[0]]
                fname_np_mask = os.path.join(self.root_np_path,os.path.basename(file) + '_' + subtask_item +'_mask.npy')
                np.save(fname_np_mask,mask)
                if mode == 'train':
                    target = loadmat(target_path)
                    target = target[list(target.keys())[0]]
                    if not target.shape==data.shape:
                        raise ValueError(f'target shape {target.shape} is not equal to data shape {data.shape}.')
                    if not target.shape[:2] == mask.shape:
                        raise ValueError(f'target image shape {target.shape[:2]} is not equal to mask shape {mask.shape}.')
                for iter_slice in range(data.shape[-2]):
                    for iter_dyn in range(data.shape[-1]):
                        data_1s1d = data[...,iter_slice,iter_dyn]
                        raw_sample = []
                        fname_np = os.path.join(self.root_np_path,os.path.basename(file) + '_' + subtask_item +'_s' + str(iter_slice + 1) + '_d' + str(iter_dyn + 1) + '.npy')
                        raw_sample.append(fname_np)
                        raw_sample.append(fname_np_mask)
                        if mode == 'train':
                            target_1s1d = target[...,iter_slice,iter_dyn]
                            fname_np_target = os.path.join(self.root_np_path,os.path.basename(file) + '_' + subtask_item +'_s' + str(iter_slice + 1) + '_d' + str(iter_dyn + 1) + '_target.npy')
                            raw_sample.append(fname_np_target)
                            np.save(fname_np_target,target_1s1d)
                        self.raw_samples.append(raw_sample)
                        np.save(fname_np,data_1s1d)
                        

        #save to dataset_cache file
        if Path(self.dataset_cache_file).exists():
            with open(self.dataset_cache_file,'rb') as cache_f:
                dataset_cache = pickle.load(cache_f)
        else:
            dataset_cache = {}
        dataset_cache[self.root_np_path] = self.raw_samples
        logging.info(f'Saving dataset cache to {self.dataset_cache_file}.')
        with open(self.dataset_cache_file,'wb') as cache_f:
            pickle.dump(dataset_cache,cache_f)

    def shapeSurvey(self):
        #remove data to path
        files = list(Path(os.path.join(self.root_path,self.acc_str)).iterdir())
        data_dict = dict()
        for file in sorted(files):#[files[0]]
            for subtask_item in self.subtask_list:
                #add postfix of .mat for given mat file name
                item_dict = dict()
                fname = Path(os.path.join(file,subtask_item + '.mat'))
                if fname.exists():
                    data = loadmat(fname)
                    item_dict[list(data.keys())[0]] = data[list(data.keys())[0]].shape
                else:
                    continue
                mask_path = Path(os.path.join(file,subtask_item +'_mask' + '.mat'))
                target_path = Path(os.path.join(self.root_path,self.target_str,os.path.basename(file),subtask_item + '.mat'))
                mask = loadmat(mask_path)
                item_dict[list(mask.keys())[0]] = mask[list(mask.keys())[0]].shape
                if mode == 'train':
                    target = loadmat(target_path)
                    item_dict[list(target.keys())[0]] = target[list(target.keys())[0]].shape
                data_dict[os.path.basename(Path(file)) + '_' + subtask_item] = item_dict
        pd.DataFrame(data_dict).transpose().to_excel(os.path.join(self.root_path,self.acc_str + '.xlsx'))
                    


if __name__ == "__main__":
    root = 'X:\CMRxRecon\MICCAIChallenge2023\ChallengeData'
    dataset_cache_file = 'X:\CMRxRecon\MICCAIChallenge2023\ChallengeDataCache\dataset_cache.pkl'
    challenges = ["SingleCoil", "MultiCoil"]
    tasks = ["Cine"]
    subtasks = ['sax','lax','all']
    accelerations = [4,8,10]
    modes = ['train','validation']
    for challenge in challenges:
        for task in tasks:
            for subtask in subtasks:
                for acceleration in accelerations:
                    for mode in modes:
                        CMRxReconPreprocessing(root,
                                            challenge = challenge,
                                            task = task,
                                            subtask = subtask,
                                            acceleration = acceleration,
                                            mode = mode,
                                            dataset_cache_file = dataset_cache_file).convetMat2NpwithCache()#shapeSurvey()
    # with open(dataset_cache_file,'rb') as cache_f:
    #     dataset_cache = pickle.load(cache_f)
    # print('dirty trick')
