import numpy as np
from scipy.spatial.distance import cdist

def dice_score(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Dice coefficient between two binary arrays.

    The Dice coefficient is a similarity measure commonly used in image segmentation tasks.

    Args:
        data1 (np.ndarray): The first binary array.
        data2 (np.ndarray): The second binary array.

    Returns:
        float: The Dice coefficient between the two binary arrays.

    Raises:
        ValueError: If the shapes of the input arrays are not equal.

    Example:
        >>> data1 = np.array([[0, 0, 1, 1, 0]])
        >>> data2 = np.array([[0, 1, 1, 0, 0]])
        >>> dice_score(data1, data2)
        0.6666666666666666
    """
    if not np.array_equal(data1.shape, data2.shape):
        raise ValueError("The shapes of two arrays must be equal. data1 shape: {}, data2 shape: {}".format(data1.shape, data2.shape))

    intersection = np.sum(data1 * data2)
    dice = (2.0 * intersection) / (np.sum(data1) + np.sum(data2))
    return dice

def assd_score(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Calculate the Average Symmetric Surface Distance (ASSD) between two binary arrays.

    The ASSD measures the average distance between corresponding points on the surfaces
    represented by the binary arrays.

    Args:
        data1 (np.ndarray): The first binary array representing a surface.
        data2 (np.ndarray): The second binary array representing a surface.

    Returns:
        float: The Average Symmetric Surface Distance between the two surfaces.

    Raises:
        ValueError: If the shapes of the input arrays are not equal.

    Example:
        >>> data1 = np.array([[0, 0, 0, 0, 0],
        ...                    [0, 1, 1, 1, 0],
        ...                    [0, 1, 0, 1, 0],
        ...                    [0, 1, 1, 1, 0],
        ...                    [0, 0, 0, 0, 0]])
        >>> data2 = np.array([[0, 0, 0, 0, 0],
        ...                    [0, 0, 1, 0, 0],
        ...                    [0, 1, 1, 1, 0],
        ...                    [0, 0, 1, 0, 0],
        ...                    [0, 0, 0, 0, 0]])
        >>> assd_score(data1, data2)
        1.5
    """
    if not np.array_equal(data1.shape, data2.shape):
        raise ValueError("The shapes of two arrays must be equal. data1 shape: {}, data2 shape: {}".format(data1.shape, data2.shape))

    # Convert binary arrays to sets of coordinates
    coordinates1 = np.array(np.where(data1)).T
    coordinates2 = np.array(np.where(data2)).T

    # Calculate pairwise distances between points on the surfaces
    distances_1_to_2 = cdist(coordinates1, coordinates2)
    distances_2_to_1 = cdist(coordinates2, coordinates1)

    # Calculate minimum distances from each point on one surface to the other
    min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)

    # Average the minimum distances to get the ASSD score
    assd = (np.mean(min_distances_1_to_2) + np.mean(min_distances_2_to_1)) / 2.0
    return assd

if __name__ == '__main__':
    from data import *
    gt_path = r"E:\dataset\seg10\ss_mask_new\irb82_0034.nii.gz"
    inf_path = r"C:\Users\Neurophet\Downloads\irb82_0034.nii.gz"

    gt_img = load_nii(gt_path)
    inf_img = load_nii(inf_path)

    gt_data = nifti_to_numpy(gt_img)
    inf_data = nifti_to_numpy(inf_img)

    print(dice_score(gt_data, inf_data))
    print(assd_score(gt_data, inf_data))






