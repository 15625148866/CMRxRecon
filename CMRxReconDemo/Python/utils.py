import numpy as np
import os
import mat73
import scipy
import matplotlib.pyplot as plt
from typing import Union, List,Tuple

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
                reconImg = ChallengeRecon(kspace[:,:,:,sz//2], type, reconType, imgShow)
                img4ranking = crop(np.abs(reconImg),[np.round(sx/3),np.round(sy/2),2,3]).astype(np.float32)
            else:
                reconImg = ChallengeRecon(kspace[:,:,:,sz//2], type, reconType, imgShow)
                img4ranking = crop(np.abs(reconImg),[np.round(sx/3),np.round(sy/2),2,t]).astype(np.float32)

            #mkdir for saving
            saveFilePath = os.path.join(savePath,fileName)
            if not os.path.exists(saveFilePath):
                os.mkdir(saveFilePath)
            assert os.path.isdir(saveFilePath)
            mdict = {"img4ranking":img4ranking}
            scipy.savemat(os.path.join(saveFilePath,fileType+'.mat'),mdict)
        
        print(str(AFtype[ind0]) + 'reconstructed successfully!')

def ChallengeRecon(kspace: np.ndarray,
                   type: int,
                   reconType: int,
                   imgShow: int)->np.ndarray:
    """
    % kspace: complex images with the dimensions (sx,sy,sc,sz,t/w)
    % -sx: matrix size in x-axis
    % -sy: matrix size in y-axis
    % -sc: coil array number
    % -sz: slice number (short axis view); slice group (long axis view)
    % -t/w: time frame/weighting

    % type = 0 means full kspace data
    % type = 1 means subsampled data

    % reconType = 0: perform zero-filling recon
    % reconType = 1: perform GRAPPA recon
    % reconType = 2: perform SENSE recon
    % reconType = 3: perform both GRAPPA and SENSE recon

    % imgShow = 0: ignore image imshow
    % imgShow = 1: image imshow
    """
    if type == 0:
        kspace_full = kspace
        sx,sy,scc,sz,nPhase = kspace_full.shape
        img_full = np.zeros(sx,sy,scc,sz,nPhase)
        img_full_sos = np.zeros(sx,sy,sz,nPhase)
        for ind1 in range(sz):
            for ind2 in range(nPhase):
                img_full[:,:,:,ind1,ind2] = ifft2c(kspace_full[:,:,:,ind1,ind2])
                img_full_sos[:,:,ind1,ind2] = sos(img_full[:,:,:,ind1,ind2])
        if imgShow == 1:
            plt.figure()
            plt.imshow(np.abs(img_full_sos[:,:,0,0]),vmin=0.0,vmax=0.001)
            plt.show(block=False)
        recon = img_full_sos
    else:
        #load data
        ncalib = 24
        kspace_sub = kspace
        sx,sy,scc,sz,nPhase = kspace.shape
        kspace_cal = np.zeros((sx,ncalib,scc,sz,nPhase))
        img_zf = np.zeros((sx,sy,scc,sz,nPhase))
        img_sos = np.zeros(sx,sy,sz,nPhase)
        #generate calibration data
        for ind2 in range(nPhase):
            kspace_calb = crop(kspace_sub[:,:,:,:,1],(sx,ncalib,scc,sz))
            kspace_cal[:,:,:,:,ind2] = kspace_calb
        #perform ZF recon
        if reconType == 0:
            for ind1 in range(sz):
                for ind2 in range(nPhase):
                    img_zf[:,:,:,ind1,ind2] = ifft2c(kspace_sub[:,:,:,ind1,ind2])
                    img_sos[:,:,ind1,ind2] = sos(img_zf[:,:,:,ind1,ind2])
                print(str(ind1) + '/' + str(ind2) + ' completed!')
            if imgShow == 1:
                plt.figure()
                plt.imshow(np.abs(img_full_sos[:,:,0,0]),vmin=0.0,vmax=0.001)
                plt.show(block=False)
            recon = img_sos
        #perform GRAPPA recon
        if reconType == 1 | reconType == 3:
            img_grappa = np.zeros((sx,sy,scc,sz,nPhase))
            kspace_grappa = np.zeros((sx,sy,scc,sz,nPhase))
            img_grappa_sos = np.zeros((sx,sy,sz,nPhase))
            for inn1 in range(sz):
                for ind2 in range(nPhase):
                    kspace_grappa[:,:,:,ind1,ind2], img_grappa[:,:,:,ind1,ind2] = myGRAPPA()
                    img_grappa_sos[:,:,ind1,ind2] = sos(img_grappa[:,:,:,ind1,ind2])
                print(str(ind1) + '/' + str(ind2) + ' completed!')
            if imgShow == 1:
                plt.figure()
                plt.imshow(np.abs(img_grappa_sos[:,:,0,0]),vmin=0.0,vmax=0.001)
                plt.show(block=False)
            recon = img_grappa_sos
        # perform SENSE recon
        if reconType == 2:
            img_sense = np.zeros((sx,sy,sz,nPhase))
            kspace_sense = np.zeros((sx,sy,scc,sz,nPhase))
            #perfomr sense recon
            for ind1 in range(sz):
                for ind2 in range(nPhase):
                    kspace_sense[:,:,:,ind1,ind2], img_sense[:,:,:,ind1,ind2] = mySENSE()
                print(str(ind1) + '/' + str(ind2) + ' completed!')
            if imgShow == 1:
                plt.figure()
                plt.imshow(np.abs(img_sense[:,:,0,0]),vmin=0.0,vmax=0.001)
                plt.show(block=False)
            recon = img_sense
        return recon


if __name__ == '__main__':
    x = np.arange(20).reshape((5,4))
    print(x)
    print(crop(x,(3,2)))
    print(pad(crop(x,[3,2]),(5,4)))


    
