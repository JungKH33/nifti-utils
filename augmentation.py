import numpy as np
import torchio as tio

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.orientations import axcodes2ornt
from nibabel.orientations import ornt_transform

def reorient(input_img: Nifti1Image, target_orientation: tuple[str, str, str] = ('R', 'A', 'S')) -> Nifti1Image:
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

def reshape(input_img: Nifti1Image, target_shape):

    transformed_subject = tio.Resample(target_shape)(input_img)

    transformed_image = transformed_subject['image']

    return transformed_image

