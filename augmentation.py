from __future__ import annotations

import numpy as np
import torchio as tio

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.orientations import axcodes2ornt, ornt_transform


def reorient(input_img: Nifti1Image, target_orientation: str| tuple[str, str, str] = ('R', 'A', 'S')) -> Nifti1Image:
    """
    Change the orientation of a NIfTI image to the specified target orientation.
    Target orientation string or tuple must consist of "R" or "L", "A" or "P", and "I" or "S" in any order.

    Parameters:
        input_img (nibabel.nifti1.Nifti1Image): The input NIfTI image object.
        target_orientation (tuple, optional): The target orientation as a tuple of three characters
            representing the axes (e.g., ('R', 'A', 'S') for Right, Anterior, Superior). Default is ('R', 'A', 'S').

    Returns:
        nibabel.nifti1.Nifti1Image: The NIfTI image with the orientation changed to the target orientation.
    """
    current_orientation = nib.io_orientation(input_img.affine)
    target_orientation = axcodes2ornt(target_orientation)

    transform = ornt_transform(current_orientation, target_orientation)
    transformed_image = input_img.as_reoriented(transform)

    return transformed_image
def resize(input_img: Nifti1Image, target_shape: int | tuple[int, int, int]) -> Nifti1Image:
    """
    Resize the input image to the target shape.

    Args:
        input_img (Nifti1Image): Input image to be resized.
        target_shape (tuple of int): Target shape of the output image in (x, y, z) format.

    Returns:
        Nifti1Image: Resized image with the target shape.
    """
    transformed_image = tio.Resize(target_shape)(input_img)
    return transformed_image
def resample_iso(input_img: Nifti1Image, target_iso: float | tuple[float, float, float]) -> Nifti1Image:
    """
    Resample the input image to the target isotropic resolution.

    Args:
        input_img (Nifti1Image): Input image to be resampled.
        target_iso (tuple of float): Target isotropic resolution in millimeters.

    Returns:
        Nifti1Image: Resampled image with the target isotropic resolution.
    """
    transformed_image = tio.Resize(target_iso)(input_img)
    return transformed_image
