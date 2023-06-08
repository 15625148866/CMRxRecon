#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 10 00:20:05 2023
Load .mat data using python

@author: c.y.wang
"""


import os
import numpy as np
from loadFun import loadmat
import matplotlib.pyplot as plt

data_dir = 'X:\CMRxRecon\MICCAIChallenge2023\ChallengeData\SingleCoil\Cine\TrainingSet\AccFactor04\P001'
mat_file = os.path.join(data_dir, 'cine_lax.mat')

data = loadmat(mat_file)
print('dirty trick')
data = data[list(data.keys())[0]]
data = np.array(data)
plt.figure()
plt.imshow(np.abs(data['real'] + 1j*data['imag'])[1,1,:,:])
plt.show(block=False)