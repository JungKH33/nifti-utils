from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt

def plot_image(input_image: nib.Nifti1Image, display_mode: str = 'ortho', cmap: str = 'gray') -> None:
    """
    Display a NIfTI image using Nilearn's plotting capabilities.

    Parameters:
    input_image (nibabel.nifti1.Nifti1Image): The NIfTI image to be displayed.
    display_mode (str): The display mode for the image. Can be 'ortho', 'x', 'y', or 'z' (default: 'ortho').
    cmap (str): The colormap to use for displaying the image (default: 'gray').

    Returns:
    None
    """
    plotting.plot_anat(input_image, display_mode=display_mode, cmap=cmap)
    plt.show()


def plot_image_with_mask(input_image: nib.Nifti1Image, mask_image: nib.Nifti1Image, display_mode: str = 'ortho', cmap: str = 'gray') -> None:
    """
    Display a NIfTI image with an overlay mask using Nilearn's plotting capabilities.

    Parameters:
    input_image (nibabel.nifti1.Nifti1Image): The background image to be displayed.
    mask_image (nibabel.nifti1.Nifti1Image): The binary mask to be overlayed on the background image.
    display_mode (str): The display mode for the image. Can be 'ortho', 'x', 'y', or 'z' (default: 'ortho').
    cmap (str): The colormap to use for displaying the background image (default: 'gray').

    Returns:
    None
    """
    plotting.plot_roi(roi_img=mask_image, bg_img=input_image, display_mode=display_mode, cmap=cmap, alpha=0.5)
    plt.show()