import numpy as np
import nibabel as nib

def load_nii(file_path: str) -> np.ndarray:
    """Load a NIfTI file and return the image data array.

    Parameters:
        file_path (str): The file path to the NIfTI file.

    Returns:
        np.ndarray: The image data array.
    """
    # Load the NIfTI file
    img = nib.load(file_path)
    data = img.get_fdata()
    return img, data
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
