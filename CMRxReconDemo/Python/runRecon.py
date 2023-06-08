import numpy as np
import os
import mat73
from typing import Union

def runRecon(basePath: str,
             mainSavePath: str,
             coilInfo: str,
             setName: str,
             fileType: str,
             AFtype: int,
             AFname: str,
             type: int = 0,
             reconType: int = 0,
             imgShow: int = 0
             )->np.ndarray:
    """
    Python runRecon convert from matlab in CMRxRecon

    %% parameter meaning
    % type = 0 means full kspace data
    % type = 1 means subsampled data

    % reconType = 0: perform zero-filling recon
    % reconType = 1: perform GRAPPA recon
    % reconType = 2: perform SENSE recon
    % reconType = 3: perform both GRAPPA and SENSE recon

    % imgShow = 0: ignore image imshow
    % imgShow = 1: image imshow

    % filetype: 'cine_lax', 'cine_sax', 'T1map', 'T2map'

    """
    #set name
    if fileType in ['cine_lax','cine_sax']:
        modalityName = 'Cine'
    else:
        modalityName = 'Mapping'

    #run for different Acc factors
    for ind0 in range(3):
        mainDataPath = basePath + coilInfo + modalityName + setName + AFtype[ind0]
        savePath = mainSavePath + coilInfo + modalityName + setName + AFtype[ind0]
        fileList = dir(mainDataPath)
        numFile = len(fileList)
        #runing all patients
        for fileName in fileList:
            dataPath = os.path.join(mainDataPath,os.path.join(fileName, fileType + '.mat'))
            data = mat73.load(dataPath)
            variable_name = list[data.key()][0]
            kspace = data[variable_name]
            # to reduce the computing burden and space, we only evaluate the central 2 slices
            # For cine: use the first 3 time frames for ranking!
            # For mapping: we need all weighting for ranking!
            sx,sy,_,sz,t = kspace.shape
            if fileType in ['cine_lax','cine_sax']:
                reconImg = ChallengeRecon(kspace(:,:,:,sz//2), type, reconType, imgShow)

            else:

