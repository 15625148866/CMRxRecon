import numpy as np
import runRecon


#set info
coilInfo = 'MultiCoil/'
setName = 'ValidationSet/'
AFtype = ['AccFactor04','AccFactor08','AccFactor10']
AFname = ['kspace_sub04','kspace_sub08','kspace_sub10']
#
basePath = '.'
mainSavePath = '.'

"""
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

#SENSE recon
type = 1
reconType = 0
imgShow = 1

# long axis cine
runRecon(basePath,mainSavePath,coilInfo,setName,'cine_lax',AFtype,AFname,type,reconType,imgShow); 
# short axis cine
runRecon(basePath,mainSavePath,coilInfo,setName,'cine_sax',AFtype,AFname,type,reconType,imgShow); 
# T1 mapping
runRecon(basePath,mainSavePath,coilInfo,setName,'T1map',AFtype,AFname,type,reconType,imgShow); 
# T2 mapping
runRecon(basePath,mainSavePath,coilInfo,setName,'T2map',AFtype,AFname,type,reconType,imgShow); 