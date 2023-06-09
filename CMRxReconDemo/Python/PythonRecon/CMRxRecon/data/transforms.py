import torch
import numpy as np
from typing import Tuple,Union,Optional,NamedTuple
from CMRxRecon.utils.math_utils import *

class UnetSample(NamedTuple):
    """
    A subsampled image for U-Net reconstruction.

    Args:
        image: Subsampled image after inverse FFT.
        target: The target image (if applicable).
        mean: Per-channel mean values used for normalization.
        std: Per-channel standard deviations used for normalization.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
    """

    image: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    # slice_num: int
    # max_value: float

class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
    ):
        """
        Args:
            which_challenge: Challenge from ("SingleCoil", "MultiCoil").
        """
        if which_challenge not in ("SingleCoil", "MultiCoil"):
            raise ValueError("Challenge should either be 'SingleCoil' or 'MultiCoil'")
        self.which_challenge = which_challenge

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        fname: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A tuple containing, zero-filled input image, the reconstruction
            target, the mean used for normalization, the standard deviations
            used for normalization, the filename, and the slice number.
        """
        kspace_torch = to_tensor(kspace)
        mask_torch = to_tensor(mask)

        # apply mask
        masked_kspace_torch = apply_mask(kspace_torch,mask_torch,self.which_challenge)

        #if multiCoil, transpose
        if self.which_challenge == 'MultiCoil':
            kspace_torch = masked_kspace_torch.movedim(-2,0) #change channel to slice dims

        # inverse Fourier transform to get zero filled solution
        image_sub_torch_all_chan = ifft2c_torch(kspace_torch)

        # pad to [512,256] first input to correct size
        pad_size = (512,256)
        image_sub_torch_all_chan = complex_center_pad(image_sub_torch_all_chan, pad_size)
        # crop_size = (kspace_torch.shape[-3]//3,kspace_torch.shape[-2]//2) # [H,W,complex]

        # image_sub_torch = complex_center_crop(image_sub_torch, crop_size)

        # absolute value
        image_sub_torch_comb = abs_torch(image_sub_torch_all_chan)

        # apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == "MultiCoil":
            image_sub_torch_comb = rss_torch(image_sub_torch_comb,0)

        # normalize input
        image_sub_torch_comb, mean_sub, std_sub = normalize_instance(image_sub_torch_comb, eps=1e-11)
        image_sub_torch_comb = image_sub_torch_comb.clamp(-6, 6)

        # normalize target
        if target is not None:
            target_torch = to_tensor(target)
            if self.which_challenge == 'MultiCoil':
                target_torch = target_torch.movedim(-2,0)
            image_full_torch_all_chan = ifft2c_torch(target_torch)
            image_full_torch_all_chan = complex_center_pad(image_full_torch_all_chan, pad_size)
            image_full_torch_comb = abs_torch(image_full_torch_all_chan)
            if self.which_challenge == 'MultiCoil':
                image_full_torch_comb = rss_torch(image_full_torch_comb,0)
            image_full_torch_comb = normalize(image_full_torch_comb, mean_sub, std_sub, eps=1e-11)
            image_full_torch_comb = image_full_torch_comb.clamp(-6, 6)
        else:
            image_full_torch_comb = torch.Tensor([0])

        return UnetSample(
            image=image_sub_torch_comb,
            target=image_full_torch_comb,
            mean=mean_sub,
            std=std_sub,
            fname = fname
        )