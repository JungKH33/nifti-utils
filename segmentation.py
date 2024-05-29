import os
from typing import Optional, Tuple
import numpy as np

from skimage.segmentation import find_boundaries
from skimage.measure import label, regionprops
from sklearn.cluster import DBSCAN, KMeans

from scipy.ndimage import binary_dilation, measurements

import data

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

def get_borders(input_data: np.ndarray, labels: list = None, connectivity = 1, mode: str = 'inner') -> np.ndarray:
    """Extract the borders of specified labels in an image.

    Parameters:
        input_data (np.ndarray): The input image data array.
        labels (list): List of labels for which borders are to be extracted. If None, borders for all unique labels will be extracted.
        mode (str): Specifies how boundaries should be identified. Possible values are 'inner' and 'outer'.
                    - 'inner': Boundaries are identified within the labeled regions.
                    - 'outer': Boundaries are identified between the labeled regions and the background.

    Returns:
        np.ndarray: The image with the extracted borders assigned the target label.
    """

    # Initialize an empty array to store the borders
    borders_combined = np.zeros_like(input_data)

    # If labels is None, get all unique labels from the input data
    if labels is None:
        # Find the borders between the specified label and other labels
        borders = find_boundaries(input_data, connectivity=connectivity, mode='inner')
        return borders

    else:
        # Find and combine borders for each specified label
        for label in labels:
            # Create a mask where specified label is True and all others are False
            mask_specified = (input_data == label)

            # Find the borders between the specified label and other labels
            borders = find_boundaries(mask_specified, connectivity= connectivity, mode='inner')

            # Add the borders to the combined array
            borders_combined = np.logical_or(borders_combined, borders)

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
        dbscan = DBSCAN(eps= 1, min_samples= 1)
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

def get_connected_components(input_data: np.ndarray, connectivity: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    Get connected components in the input data.

    Parameters:
    input_data (np.ndarray): The input data containing regions.
    connectivity (Optional[int]): Maximum number of orthogonal hops to consider a pixel/voxel as a neighbor.
        Accepted values are ranging from 1 to input.ndim. If None, a full connectivity of input.ndim is used.

    Returns:
    Tuple[np.ndarray, int]: A tuple containing a mask where each connected component is labeled with a unique integer,
    and the number of objects found.
    """
    # Find connected components in the input data
    connected_component_mask, num_objects = label(input_data, return_num= True, connectivity= connectivity)

    return connected_component_mask, num_objects

def get_bbox(input_data: np.ndarray) -> dict:
    """
    Create bounding boxes for each label in the input data.

    Parameters:
    input_data (np.ndarray): The input data containing labeled regions.

    Returns:
    dict: A dictionary where keys are labels and values are bounding boxes,
          each bounding box is represented as a tuple of two 3D coordinates: ((min_x, min_y, min_z), (max_x, max_y, max_z)).
    """
    bounding_boxes = {}

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

        # Store the bounding box coordinates in the dictionary
        bounding_boxes[label] = ((min_x, min_y, min_z), (max_x, max_y, max_z))

    return bounding_boxes

def get_regions(input_data: np.ndarray, spacings: tuple = None) -> dict:
    """
    Calculate region properties from a labeled image array and return them in a dictionary.

    Parameters:
        input_data (np.ndarray): The input labeled image array where each region is identified by a unique label.
        spacings (tuple, optional): A tuple of three float values representing the voxel spacing along each axis (z, y, x).
                                    If provided, the volume with spacing will be calculated for each region.

    Returns:
        dict: A dictionary where the keys are the region labels and the values are dictionaries containing properties
              of each region, such as volume, bounding box volume, bounding box coordinates, and volume with spacing (if spacings are provided).
    """
    region_dict = {}
    regions = regionprops(input_data)

    if spacings is not None:
        voxel_volume = spacings[0] * spacings[1] * spacings[2]

    for region in regions:
        region_label = region.label
        region_dict[region_label] = {
            'volume': region.area,
            'bbox_volume': region.bbox_area,
            'bbox': region.bbox
        }

        if spacings is not None:
            region_dict[region_label]['volume_spacing'] = region.bbox_area * voxel_volume

    return region_dict

def remove_small_clusters(image, threshold):
    labeled_array, num_features = label(image, return_num= True)
    for i in range(1, num_features + 1):
        cluster = (labeled_array == i)
        if np.sum(cluster) < threshold:
            image[cluster] = 0
    return image

if __name__ == "__main__":
    from data import *
    import experimental
    from sklearn.cluster import DBSCAN

    dbscan = True

    input_dir = r"E:\dataset_final\ss\swi\ground_truth"
    data_dict = {}

    for filename in os.listdir(input_dir):
        print(filename)
        if filename == 'irb83_0081.nii.gz':
            break
        input_path = os.path.join(input_dir, filename)
        img = load_nii(input_path)
        img_np = nifti_to_numpy(img, np.int32)

        spacing = extract_nifti_info(img)['spacings']
        volume = get_regions(img_np, spacing)[1]['volume_spacing']

        clusters, num_clusters = label(img_np, return_num=True, connectivity= None)

        if dbscan == True:
            coords = np.column_stack(np.where(img_np != 0))
            dbscan = DBSCAN(eps=1, min_samples=1)
            clusters = dbscan.fit_predict(coords)
            unique_labels = np.unique(clusters)
            num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)

        data_dict[filename] = (volume, num_clusters)
    ####
    import matplotlib.pyplot as plt

    data_dict = dict(sorted(data_dict.items(), key=lambda item: item[1][0]))

    # Separate the filenames, volumes, and num_clusters
    filenames = list(data_dict.keys())
    volumes = [v[0] for v in data_dict.values()]
    num_clusters = [v[1] for v in data_dict.values()]

    # Define the positions for the bars
    x = np.arange(len(filenames))
    width = 0.35  # Width of the bars

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot volumes on the primary y-axis
    bars1 = ax1.bar(x - width / 2, volumes, width, label='Volume', color='blue', edgecolor='black')
    ax1.set_xlabel('Filename')
    ax1.set_ylabel('Volume', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot number of clusters on the secondary y-axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, num_clusters, width, label='Number of Clusters', color='orange', edgecolor='black')
    ax2.set_ylabel('Number of Clusters', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Add some text for title and axes ticks
    plt.title('Volume and Number of Clusters by Filename')
    ax1.set_xticks(x)
    ax1.set_xticklabels(filenames, rotation=45, ha='right', fontsize=8)


    # Attach a text label above each bar, displaying its height
    def autolabel(ax, bars):
        """Attach a text label above each bar in `bars`, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)


    #autolabel(ax1, bars1)
    #autolabel(ax2, bars2)

    # Adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()


    import sys
    sys.exit()
    input_dir = r"E:\dataset\seg10\ss_input_reorient"
    mask_dir = r"E:\dataset\seg10\ss_mask_new"
    save_dir = r"C:\github\nii-utils\src"

    for filename in os.listdir(input_dir):

        input_path = os.path.join(input_dir, filename)
        mask_path = os.path.join(mask_dir, filename)
        save_path = os.path.join(save_dir, filename)

        img = load_nii(input_path)
        mask_img = load_nii(mask_path)
        mask_data = nifti_to_numpy(mask_img)
        # clustered_data = cluster_labels(mask_data)
        clustered_data, num_objects = get_connected_components(mask_data, connectivity= 1)
        bounding_box = get_bbox(clustered_data)
        # regions = get_regions(clustered_data)
        # clustered_data = experimental.draw_bounding_boxes(clustered_data,bounding_box)
        save_nii(save_path, clustered_data)
