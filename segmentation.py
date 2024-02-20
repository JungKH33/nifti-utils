import numpy as np
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation

def find_labels(input_data: np.ndarray) -> list:
    """Find all unique labels present in a mask.

    Parameters:
        input_data (np.ndarray): The input mask.

    Returns:
        list: A list of unique labels present in the mask.
    """
    # Find unique non-zero elements in the mask
    unique_labels = np.unique(input_data)

    # Convert the unique labels to a list
    label_list = list(unique_labels)

    return label_list
def merge_labels(input_data: np.ndarray, labels: list, target_label: tuple = (1, 0)) -> np.ndarray:
    """Merge specified labels in the input data array to target labels.

    Parameters:
        input_data (np.ndarray): The input data array.
        labels (list): The list of labels to be merged.
        target_label (tuple): A tuple specifying the target labels.
            If target_label[1] is None, labels will be merged to target_label[0].
            If target_label[1] is not None, labels present in input_data will be merged to target_label[0],
            while labels not present in input_data will be merged to target_label[1].
            Default is (1, None).

    Returns:
        np.ndarray: The merged labels array.
    """

    if target_label[1] is None:
        # Set the mask values to 1 for specific labels
        merged_labels = np.where(np.isin(input_data, labels), target_label[0], input_data)
    else:
        merged_labels = np.where(np.isin(input_data, labels), target_label[0], target_label[1])

    return merged_labels

def get_borders(input_data: np.ndarray, labels: list, thickness: int = 1) -> np.ndarray:
    """Extract the borders of a specified label in an image.

    Parameters:
        input_data (np.ndarray): The input image data array.
        labels (list): The list of labels to be merged.
        thickness (int): The thickness of the borders to be extracted. Default is 3.

    Returns:
        np.ndarray: The binary mask containing the extracted borders.
    """
    # Initialize an empty mask for the specified labels
    mask_labels = np.zeros_like(input_data, dtype=bool)

    # Create a mask for each specified label and combine them
    for label in labels:
        mask_labels |= (input_data == label)

    # Find the borders between the specified labels and other labels
    borders = find_boundaries(mask_labels, connectivity=1, mode='inner')

    # Dilate the borders to increase thickness
    borders = binary_dilation(borders, iterations=thickness)

    return borders
def add_padding(input_data: np.ndarray, labels: list, padding_size: int = 1) -> np.ndarray:
    """Add padding to specified labels in an image.

    Parameters:
        input_data (np.ndarray): The input image data array.
        labels (list): The list of labels to which padding will be added.
        padding_size (int): The size of the padding to be added. Default is 1.

    Returns:
        np.ndarray: The image data array with padding added to the specified labels.
    """
    padded_img = np.zeros_like(input_data)
    for label in labels:
        # Create a mask for the current label
        label_mask = (input_data == label)
        # Dilate the mask to add padding
        padded_label_mask = binary_dilation(label_mask, iterations=padding_size)
        # Set the pixels in the padded mask to the current label
        padded_img[padded_label_mask] = label

    return padded_img