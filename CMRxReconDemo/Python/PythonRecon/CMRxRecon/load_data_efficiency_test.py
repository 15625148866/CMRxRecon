import numpy as np
import mat73
import h5py
import scipy.io as sio
import pymatreader
import time
from utils.math_utils import sos
import os

def loadmat_mat73(filename):
    """
    Load Matlab v7.3 format .mat file using mat73.
    """
    data = mat73.loadmat(filename)
    # data = sio.loadmat(filename)
    # data = pymatreader.read_mat(filename)
    # with h5py.File(filename, 'r') as f:
    #     data = {}
    #     for k, v in f.items():
    #         if isinstance(v, h5py.Dataset):
    #             data[k] = v[()]
    #         elif isinstance(v, h5py.Group):
    #             data[k] = loadmat_group(v)
    return data

def loadmat_scipy(filename):
    """
    Load Matlab v7.3 format .mat file using scipy.io.
    """
    # data = mat73.loadmat(filename)
    data = sio.loadmat(filename)
    # data = pymatreader.read_mat(filename)
    # with h5py.File(filename, 'r') as f:
    #     data = {}
    #     for k, v in f.items():
    #         if isinstance(v, h5py.Dataset):
    #             data[k] = v[()]
    #         elif isinstance(v, h5py.Group):
    #             data[k] = loadmat_group(v)
    return data

def loadmat_pymatreader(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    # data = mat73.loadmat(filename)
    # data = sio.loadmat(filename)
    data = pymatreader.read_mat(filename)
    # with h5py.File(filename, 'r') as f:
    #     data = {}
    #     for k, v in f.items():
    #         if isinstance(v, h5py.Dataset):
    #             data[k] = v[()]
    #         elif isinstance(v, h5py.Group):
    #             data[k] = loadmat_group(v)
    return data

def loadmat_h5py(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
    """
    # data = mat73.loadmat(filename)
    # data = sio.loadmat(filename)
    # data = pymatreader.read_mat(filename)
    with h5py.File(filename, 'r') as f:
        data = {}
        for k, v in f.items():
            if isinstance(v, h5py.Dataset):
                data[k] = v[()]
            elif isinstance(v, h5py.Group):
                data[k] = loadmat_group(v)
    return data

def loadmat_group(group):
    """
    Load a group in Matlab v7.3 format .mat file using h5py.
    """
    data = {}
    for k, v in group.items():
        if isinstance(v, h5py.Dataset):
            data[k] = v[()]
        elif isinstance(v, h5py.Group):
            data[k] = loadmat_group(v)
    return data

if __name__ == '__main__':
    #HDD test
    print('******************TEST FOR HDD*********************')
    HDD_root = r'X:\CMRxRecon\MICCAIChallenge2023\ChallengeData\MultiCoil\Cine\TrainingSet\AccFactor04\P001'
    matfile_path = os.path.join(HDD_root,'cine_lax.mat') #100MB
    file_size = os.stat(matfile_path).st_size / (1024 * 1024) # in MegaBytes
    start = time.time()
    data_mat73 = loadmat_mat73(matfile_path)
    data_mat73 = data_mat73[list(data_mat73.keys())[0]]
    elapse_mat73 = time.time() - start
    print(f'mat73 load .matfile with size of {file_size} spend time: {elapse_mat73} s.')
    start = time.time()
    data_h5py = loadmat_h5py(matfile_path)
    data_h5py = data_h5py[list(data_h5py.keys())[0]]
    data_h5py = np.array(data_h5py['real'] + 1j*data_h5py['imag'])
    elapse_h5py = time.time() - start
    print(f'h5py load .matfile with size of {file_size} spend time: {elapse_h5py} s.')
    print(f'h5py load vs. mat73 load speed gain: {elapse_mat73 / elapse_h5py}')
    print(f'difference between h5py load and mat73 load is :' + str(np.sum(sos(data_h5py - np.transpose(data_mat73)))))
    data_npy = np.array(data_h5py)
    np.save(os.path.join(HDD_root,'test.npy'),data_npy)
    start = time.time()
    data_npy = np.load(os.path.join(HDD_root,'test.npy'))
    elapse_npy = time.time() - start
    print(f'npy load .matfile with size of {file_size} spend time: {elapse_npy} s.')
    print(f'npy load vs. mat73 load speed gain: {elapse_mat73 / elapse_npy}')
    print(f'difference between h5py load and npy load is :' + str(np.sum(sos(data_h5py - data_npy))))
    np.savez(os.path.join(HDD_root,'test.npz'),data = data_npy)
    start = time.time()
    data_npz = np.load(os.path.join(HDD_root,'test.npz'))['data']
    elapse_npz = time.time() - start
    print(f'npz load .matfile with size of {file_size} spend time: {elapse_npz} s.')
    print(f'npy load vs. mat73 load speed gain: {elapse_mat73 / elapse_npz}')
    print(f'difference between h5py load and npz load is :' + str(np.sum(sos(data_h5py - data_npz))))
    #multi-file comparison
    np.save(os.path.join(HDD_root,'test1.npy'),data_npy)
    np.save(os.path.join(HDD_root,'test2.npy'),data_npy)
    start = time.time()
    data_npy1 = np.load(os.path.join(HDD_root,'test1.npy'))
    data_npy2 = np.load(os.path.join(HDD_root,'test2.npy'))
    elapse_npy = time.time() - start
    print(f'npy load 2*.matfile with size of {file_size} spend time: {elapse_npy} s.')
    print(f'npy load vs. mat73 load speed gain: {elapse_mat73 / elapse_npy}')
    print(f'difference between h5py load and npy load is :' + str(np.sum(sos(data_h5py - data_npy1))))
    np.savez(os.path.join(HDD_root,'test.npz'),data1 = data_npy, data2 = data_npy)
    start = time.time()
    data_npz = np.load(os.path.join(HDD_root,'test.npz'))
    data_npz1 = data_npz['data1']
    data_npz2 = data_npz['data2']
    elapse_npz = time.time() - start
    print(f'npz load .matfile with size of {file_size} spend time: {elapse_npz} s.')
    print(f'npz load vs. mat73 load speed gain: {elapse_mat73 / elapse_npz}')
    print(f'difference between h5py load and npz load is :' + str(np.sum(sos(data_h5py - data_npz1))))

    #SSD test
    print('******************TEST FOR SSD*********************')
    SSD_root = r'D:\fastMRI_dataset'
    matfile_path = os.path.join(SSD_root,'cine_lax.mat') #100MB
    file_size = os.stat(matfile_path).st_size / (1024 * 1024) # in MegaBytes
    start = time.time()
    data_mat73 = loadmat_mat73(matfile_path)
    data_mat73 = data_mat73[list(data_mat73.keys())[0]]
    elapse_mat73 = time.time() - start
    print(f'mat73 load .matfile with size of {file_size} spend time: {elapse_mat73} s.')
    start = time.time()
    data_h5py = loadmat_h5py(matfile_path)
    data_h5py = data_h5py[list(data_h5py.keys())[0]]
    data_h5py = np.array(data_h5py['real'] + 1j*data_h5py['imag'])
    elapse_h5py = time.time() - start
    print(f'h5py load .matfile with size of {file_size} spend time: {elapse_h5py} s.')
    print(f'h5py load vs. mat73 load speed gain: {elapse_mat73 / elapse_h5py}')
    print(f'difference between h5py load and mat73 load is :' + str(np.sum(sos(data_h5py - np.transpose(data_mat73)))))
    data_npy = np.array(data_h5py)
    np.save(os.path.join(SSD_root,'test.npy'),data_npy)
    start = time.time()
    data_npy = np.load(os.path.join(SSD_root,'test.npy'))
    elapse_npy = time.time() - start
    print(f'npy load .matfile with size of {file_size} spend time: {elapse_npy} s.')
    print(f'npy load vs. mat73 load speed gain: {elapse_mat73 / elapse_npy}')
    print(f'difference between h5py load and npy load is :' + str(np.sum(sos(data_h5py - data_npy))))
    np.savez(os.path.join(SSD_root,'test.npz'),data = data_npy)
    start = time.time()
    data_npz = np.load(os.path.join(SSD_root,'test.npz'))['data']
    elapse_npz = time.time() - start
    print(f'npz load .matfile with size of {file_size} spend time: {elapse_npz} s.')
    print(f'npz load vs. mat73 load speed gain: {elapse_mat73 / elapse_npz}')
    print(f'difference between h5py load and npz load is :' + str(np.sum(sos(data_h5py - data_npz))))
    #multi-file comparison
    np.save(os.path.join(SSD_root,'test1.npy'),data_npy)
    np.save(os.path.join(SSD_root,'test2.npy'),data_npy)
    start = time.time()
    data_npy1 = np.load(os.path.join(SSD_root,'test1.npy'))
    data_npy2 = np.load(os.path.join(SSD_root,'test2.npy'))
    elapse_npy = time.time() - start
    print(f'npy load 2*.matfile with size of {file_size} spend time: {elapse_npy} s.')
    print(f'npy load vs. mat73 load speed gain: {elapse_mat73 / elapse_npy}')
    print(f'difference between h5py load and npy load is :' + str(np.sum(sos(data_h5py - data_npy1))))
    np.savez(os.path.join(SSD_root,'test.npz'),data1 = data_npy, data2 = data_npy)
    start = time.time()
    data_npz = np.load(os.path.join(SSD_root,'test.npz'))
    data_npz1 = data_npz['data1']
    data_npz2 = data_npz['data2']
    elapse_npz = time.time() - start
    print(f'npz load .matfile with size of {file_size} spend time: {elapse_npz} s.')
    print(f'npz load vs. mat73 load speed gain: {elapse_mat73 / elapse_npz}')
    print(f'difference between h5py load and npz load is :' + str(np.sum(sos(data_h5py - data_npz1))))


