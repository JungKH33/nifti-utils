import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image

def load_nii(file_path: str) -> Nifti1Image:
    """Load a NIfTI file and return the image data array.

    Parameters:
        file_path (str): The file path to the NIfTI file.

    Returns:
        img (nibabel.nifti1.Nifti1Image): The NIfTI image object.
    """
    # Load the NIfTI file
    img = nib.load(file_path)
    return img
def save_nii(save_path: str, input_data: np.ndarray, img_affine: np.ndarray = None) -> None:
    """Save an image data array as a NIfTI file.

    Parameters:
        save_path (str): The file path to save the NIfTI file.
        input_data (np.ndarray): The image data array.
        img_affine (np.ndarray): The affine transformation matrix.
    """

    if input_data.dtype == np.bool_:
        input_data = input_data.astype(np.uint8)

        # Save the NIfTI file
    nii_img = nib.Nifti1Image(input_data, img_affine, None)
    nib.save(nii_img, save_path)
def nifti_to_numpy(input_img: Nifti1Image) -> np.ndarray:
    """
    Convert a NIfTI image object to a NumPy array.

    Parameters:
        input_img (nibabel.nifti1.Nifti1Image): The input NIfTI image object.

    Returns:
        np.ndarray: The image data array.
    """
    # Extract image data array
    data = input_img.get_fdata()
    return data
def extract_nifti_info(input_img: Nifti1Image) -> dict:
    """
    Extract header information, data type, dimension, unique values, and orientation from a NIfTI image object.

    Parameters:
        input_img (nibabel.nifti1.Nifti1Image): The input NIfTI image object.

    Returns:
        dict: A dictionary containing:
            - 'header' (nibabel.nifti1.Nifti1Header): The header of the NIfTI image.
            - 'data_type' (numpy.dtype): The data type of the NIfTI image.
            - 'shape' (tuple): The dimensions of the NIfTI image.
            - 'unique_values' (numpy.ndarray): The unique values in the NIfTI image data.
            - 'orientation' (str): The orientation information of the NIfTI image.
    """
    nii_info = {}

    # Extract header information
    nii_info['header'] = input_img.header
    nii_info['voxel_sizes'] = nii_info['header'].get_zooms()

    # Extract data type
    nii_info['data_type'] = input_img.get_data_dtype()

    # Extract shape
    nii_info['shape'] = input_img.shape

    # Extract unique values
    nii_info['unique_values'] = np.unique(input_img.get_fdata())

    # Extract orientation
    affine = input_img.affine
    nii_info['orientation'] = nib.aff2axcodes(affine)

    return nii_info
