import numpy as np
import h5py
from typing import Union,List,Tuple,Optional
import mat73
import scipy.io as sio
import pymatreader
import torch

def crop(x: np.ndarray,
         shape: Union[List, Tuple]):
    """
    crop a 2D matrix around its center
    """
    assert len(x.shape) >= len(shape)
    if len(x.shape) != len(shape):
        shape_new = np.array(x.shape)
        shape_new[:len(shape)] = np.array(shape)
        shape = shape_new
    
    for iterDim in range(len(x.shape)):
        if shape[iterDim] > x.shape[iterDim]:
            raise ValueError(f'crop shape {shape} should be smaller than {x.shape} in dim{iterDim}.')
    #Init Crop Image
    exec('cropImg = x[' + ','.join([str((x.shape[iterDim] - shape[iterDim])//2) +':' + str((x.shape[iterDim] - shape[iterDim])//2 + shape[iterDim]) for iterDim in range(len(x.shape))])+']')
    return locals()['cropImg']

def pad(x: np.ndarray,
        shape: Union[List, tuple]):
    """
    Pad a 2D matrix around its center
    """
    assert len(x.shape) >= len(shape)
    if len(x.shape) != len(shape):
        shape_new = np.array(x.shape)
        shape_new[:len(shape)] = np.array(shape)
        shape = shape_new
    
    for iterDim in range(len(x.shape)):
        if shape[iterDim] < x.shape[iterDim]:
            raise ValueError(f'pad shape {shape} should be larger than {x.shape} in dim{iterDim}.')
    #Init Crop Image
    padImg = np.zeros(shape)
    exec('padImg[' + ','.join([str((shape[iterDim] - x.shape[iterDim])//2) +':' + str((shape[iterDim] - x.shape[iterDim])//2 + x.shape[iterDim]) for iterDim in range(len(x.shape))])+'] = x')
    return locals()['padImg']

def sos(x:np.ndarray,
        dim: int = None,
        pnorm: int=2):
    """
    Square root of sum of suqare along dim.

    Args:
        x: input nd np.array
        dim: dim for calc, default last
        pnorm: norm for sos
    """
    if dim is None:
        dim = len(x.shape) -1
    
    res = np.power(np.sum(np.abs(np.power(x, pnorm)),axis=dim),1/pnorm)
    return res

def ifft2c(x):
    """
    2D iFFT for complex data.
    """
    S = x.shape
    fctr = np.sqrt(np.prod(x.shape[:2]))
    x = np.reshape(x,(S[0],S[1],np.prod(S[2:])))
    res = np.zeros(x.shape)
    for iter in range(x.shape[2]):
        res[:,:,iter] = fctr * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x[:,:,iter])))
    res = np.reshape(res,S)

def fft2c(x):
    """
    2D FFT for complex data.
    """
    S = x.shape
    fctr = 1 / np.sqrt(np.prod(x.shape[:2]))
    x = np.reshape(x,(S[0],S[1],np.prod(S[2:])))
    res = np.zeros(x.shape)
    for iter in range(x.shape[2]):
        res[:,:,iter] = fctr * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x[:,:,iter])))
    res = np.reshape(res,S)

def fft2c_torch(data: torch.Tensor,
                dim: Optional[Tuple[int]] = (-2,-1),
                norm: Optional[str]='ortho')->torch.Tensor:
    """
    2D FFT for complex data in Torch
    """
    data = torch.view_as_complex(data)
    res = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(data),dim=dim,norm=norm))
    res = torch.view_as_real(res)
    return res

def ifft2c_torch(data: torch.Tensor,
                 dim: Optional[Tuple[int]] = (-2,-1),
                 norm: Optional[str] = 'ortho')->torch.Tensor:
    """
    2D iFFT for complex data in Torch
    """
    data = torch.view_as_complex(data)
    res = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(data),dim=dim,norm=norm))
    res = torch.view_as_real(res)
    return res

def abs_torch(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()

def rss_torch(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data**2).sum(dim))


def rss_complex_torch(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError("Tensors do not have separate complex dim.")

    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]

    return torch.stack((re, im), dim=-1)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1).sqrt()


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    return (data**2).sum(dim=-1)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


    
def run4Ranking(img, fileType):
    """
    %% this function helps you to convert your data for ranking
    % img: complex images reconstructed with the dimensions (sx,sy,sz,t/w) in
    % original size
    % -sx: matrix size in x-axis
    % -sy: matrix size in y-axis
    % -sz: slice number (short axis view); slice group (long axis view)
    % -t/w: time frame/weighting

    % img4ranking: "single" format images with the dimensions (sx/3,sy/2,2,3)
    % -sx/3: 1/3 of the matrix size in x-axis
    % -sy/2: half of the matrix size in y-axis
    % img4ranking is the data we used for ranking!!!
    """
    sx, sy, _, sz, t = img.shape
    if fileType in ['cine_lax','cine_sax']:
        img4ranking = crop(np.abs(img),[np.round(sx/3),np.round(sy/2),2,3]).astype(np.float32)
    else:
        img4ranking = crop(np.abs(img),[np.round(sx/3),np.round(sy/2),2,t]).astype(np.float32)
    
    return img4ranking

def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std

def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]

def complex_center_pad(data: torch.Tensor, shape: Tuple[int,int]) -> torch.Tensor:
    """
    Apply a center pad to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be larger than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < data.shape[-3] <= shape[0] and 0 < data.shape[-2] <= shape[1]):
        raise ValueError("Invalid shapes.")
    target_shape = list(data.shape)
    target_shape[-3] = shape[0]
    target_shape[-2] = shape[1]
    res = torch.zeros(target_shape,dtype=data.dtype)

    w_from = (shape[0] - data.shape[-3]) // 2
    h_from = (shape[1] - data.shape[-2]) // 2
    w_to = w_from + data.shape[-3]
    h_to = h_from + data.shape[-2]

    res[...,w_from:w_to,h_from:h_to,:] = data

    return res


def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y

def to_tensor(data: np.ndarray)->torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    return torch.view_as_complex(data).numpy()


def apply_mask(
    data: torch.Tensor,
    mask: torch.Tensor,
    which_challenge: str,
) -> torch.Tensor:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.

    Returns:
        masked data: Subsampled k-space data.
    """
    mask = mask.unsqueeze(-1) # append 1 at last dim
    if which_challenge == 'MultiCoil':
        mask = mask.unsqueeze(-1) #for broadcast multiply,
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data

def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]

    return mask

def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]



def loadmat(filename):
    """
    Load Matlab v7.3 format .mat file using h5py.
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