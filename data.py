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

    elif input_data.dtype == np.int64:
        input_data = input_data.astype(np.int16)

    elif input_data.dtype == np.float64:
        input_data = input_data.astype(np.float32)


        # Save the NIfTI file
    nii_img = nib.Nifti1Image(input_data, img_affine, None)
    nib.save(nii_img, save_path)


def nifti_to_numpy(input_img: nib.Nifti1Image, dtype: np.dtype = None) -> np.ndarray:
    """
    Convert a NIfTI image object to a NumPy array with an optional specified data type.

    Parameters:
        input_img (nibabel.nifti1.Nifti1Image): The input NIfTI image object.
        dtype (np.dtype, optional): The desired data type for the output array. Defaults to None.

    Returns:
        np.ndarray: The image data array with the specified data type if provided.
    """
    # Extract image data array
    data = input_img.get_fdata()

    # If a dtype is specified, convert the data to that type
    if dtype is not None:
        data = data.astype(dtype)

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
    nii_info['spacings'] = input_img.header.get_zooms()
    nii_info['origin'] = input_img.header.get_best_affine()

    # Extract data type
    nii_info['data type'] = input_img.get_data_dtype()

    # Extract shape
    nii_info['shape'] = input_img.shape

    # Extract unique values
    nii_info['unique_values'] = np.unique(input_img.get_fdata())

    # Extract orientation
    affine = input_img.affine
    nii_info['direction'] = affine[:3, :3]
    nii_info['orientation'] = nib.aff2axcodes(affine)

    return nii_info

def check_data_integrity(input_path, gt_path) -> bool:
    data_integrity = True

    input_img = load_nii(input_path)
    gt_img = load_nii(gt_path)

    input_info = extract_nifti_info(input_img)
    gt_info = extract_nifti_info(gt_img)

    input_shape = input_info['shape']
    input_orientation = input_info['orientation']
    input_direction = input_info['direction']

    gt_shape = gt_info['shape']
    gt_orientation = gt_info['orientation']
    gt_direction = gt_info['direction']

    gt_values = gt_info['unique_values']

    print()
    print("Checking data integrity for")
    print(f'input path: {input_path}  and gt path: {gt_path}')

    print(f'input shape: {input_shape}       gt shape: {gt_shape}')
    print(f'input orientation: {input_orientation}         gt orientation: {gt_orientation}')
    print(f'input direction: {input_direction}         gt direction: {gt_direction}')
    print(f'gt values: {gt_values}')

    if input_shape != gt_shape:
        print(f"Shapes of input ({input_shape}) and gt ({gt_shape}) do not match.")
        data_integrity = False

    if input_orientation != gt_orientation:
        print(f"Orientation of input ({input_orientation}) and gt ({gt_orientation}) do not match.")
        data_integrity = False

    if input_direction.any() != gt_direction.any():
        print(f"Direction of input ({input_direction}) and gt ({gt_direction}) do not match.")
        data_integrity = False

    return data_integrity

if __name__ == '__main__':
    import os
    from collections import defaultdict

    data_dir = r"E:\dataset_final\ss\swi\input"
    data_dir = r"C:\Users\Neurophet\Downloads\20240513_TechBroadcast\20240513_TechBroadcast\LABEL"

    shape_counts = defaultdict(int)
    orientation_counts = defaultdict(int)

    for filename in os.listdir(data_dir):
        print()
        print(filename)
        input_path = os.path.join(data_dir, filename)

        input_img = load_nii(input_path)
        input_info = extract_nifti_info(input_img)

        input_shape = input_info['shape']
        input_orientation = input_info['orientation']
        input_direction = input_info['direction']
        input_spacing = input_info['spacings']
        input_values = input_info['unique_values']

        print(f'input shape: {input_shape}')
        print(f'input spacing: {input_spacing}')
        print(f'input orientation: {input_orientation}')
        print(f'input direction: {input_direction}')
        print(f'unique values: {input_values}')

        shape_str = str(input_shape)
        orientation_str = str(input_orientation)

        # Update counts
        shape_counts[shape_str] += 1
        orientation_counts[orientation_str] += 1

    print("\nShape counts:")
    for shape, count in shape_counts.items():
        print(f"{shape}: {count}")

    print("\nOrientation counts:")
    for orientation, count in orientation_counts.items():
        print(f"{orientation}: {count}")