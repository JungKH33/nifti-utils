import numpy as np
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation, measurements
from sklearn.cluster import DBSCAN, KMeans

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
def change_labels(input_data: np.ndarray, labels: list, target_label: tuple = (1, 0)) -> np.ndarray:
    """Change specified labels in the input data array to target labels.

    Parameters:
        input_data (np.ndarray): The input data array.
        labels (list): The list of labels to be changed.
        target_label (tuple): A tuple specifying the target labels.
            If target_label[1] is None, labels will be changed to target_label[0].
            If target_label[1] is not None, labels present in input_data will be changed to target_label[0],
            while labels not present in input_data will be changed to target_label[1].
            Default is (1, None).

    Returns:
        np.ndarray: The changed labels array.
    """

    if target_label[1] is None:
        # Set the mask values to 1 for specific labels
        changed_labels = np.where(np.isin(input_data, labels), target_label[0], input_data)
    else:
        changed_labels = np.where(np.isin(input_data, labels), target_label[0], target_label[1])

    return changed_labels
def get_borders(input_data: np.ndarray, labels: list, target_label: int or float = 1,
                thickness: int = 1) -> np.ndarray:
    """Extract the borders of specified labels in an image and turn them into a target label.

    Parameters:
        input_data (np.ndarray): The input image data array.
        labels (list): List of labels for which borders are to be extracted.
        target_label (int or float): The label to assign to the extracted borders.
        thickness (int): The thickness of the borders to be extracted. Default is 1.

    Returns:
        np.ndarray: The image with the extracted borders assigned the target label.
    """

    # Initialize an empty array to store the borders
    borders_combined = np.zeros_like(input_data)

    # Find and combine borders for each specified label
    for label in labels:
        # Create a mask where specified label is True and all others are False
        mask_specified = (input_data == label)

        # Find the borders between the specified label and other labels
        borders = find_boundaries(mask_specified, connectivity=1, mode='inner')

        # Dilate the borders to increase thickness
        borders = binary_dilation(borders, iterations=thickness)

        # Add the borders to the combined array
        borders_combined = np.logical_or(borders_combined, borders)

    # Assign the target label to the extracted borders
    borders_combined = np.where(borders_combined, target_label, input_data)

    return borders_combined
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
def cluster_labels(input_data: np.ndarray, num_clusters: int = None) -> np.ndarray:
    """
    Cluster labels in the input data.

    Parameters:
    input_data (np.ndarray): The input data containing labels.
    num_clusters (int, optional): The number of clusters to create. If None, DBSCAN is used.
                                   Defaults to None.

    Returns:
    np.ndarray: A mask where each voxel is labeled with its cluster number.
    """
    # Get coordinates of non-background voxels
    coords = np.column_stack(np.where(input_data != 0))

    if num_clusters is None:
        # Use DBSCAN for clustering
        dbscan = DBSCAN(eps=1, min_samples=10)
        clusters = dbscan.fit_predict(coords)
    else:
        # Use KMeans for clustering
        kmeans = KMeans(n_clusters=num_clusters)
        clusters = kmeans.fit_predict(coords)

    # Initialize a mask for clustered labels
    clustered_mask = np.zeros_like(input_data)

    # Assign cluster labels to non-background voxels
    for i, (x, y, z) in enumerate(coords):
        # Add 1 to cluster labels to avoid 0 (background)
        clustered_mask[x, y, z] = clusters[i] + 1

    return clustered_mask
def get_connected_components(input_data: np.ndarray) -> np.ndarray:
    """
    Get connected components in the input data.

    Parameters:
    input_data (np.ndarray): The input data containing regions.

    Returns:
    np.ndarray: A mask where each connected component is labeled with a unique integer.
    """
    # Find connected components in the input data
    connected_component_mask, num_objects = measurements.label(input_data)

    return connected_component_mask
def create_bounding_box(input_data: np.ndarray) -> list:
    """
    Create bounding boxes for each label in the input data.

    Parameters:
    input_data (np.ndarray): The input data containing labeled regions.

    Returns:
    list: A list of bounding boxes, where each bounding box is represented
          as a tuple of two 3D coordinates: ((min_x, min_y, min_z), (max_x, max_y, max_z)).
    """
    bounding_boxes = []

    # Find unique labels in the input data
    unique_labels = np.unique(input_data)

    # Compute the number of labels, excluding the background label (0)
    num_labels = len(unique_labels) - 1

    # Iterate through each label
    for label in range(1, num_labels + 1):  # Labels start from 1
        # Find coordinates of the current label
        coords = np.argwhere(input_data == label)

        # Calculate bounding box coordinates
        min_x, min_y, min_z = np.min(coords, axis=0)
        max_x, max_y, max_z = np.max(coords, axis=0)

        # Store the bounding box coordinates
        bounding_boxes.append(((min_x, min_y, min_z), (max_x, max_y, max_z)))

    return bounding_boxes