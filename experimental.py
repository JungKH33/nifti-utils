import numpy as np
import cv2
import matplotlib.pyplot as plt


def draw_bounding_boxes(mask: np.ndarray, bounding_boxes: dict) -> np.ndarray:
    """
    Draw bounding boxes on the input image.

    Parameters:
    input_data_color (np.ndarray): The input image in color (BGR).
    bounding_boxes (dict): A dictionary where keys are labels and values are bounding boxes.

    Returns:
    np.ndarray: The input image with bounding boxes drawn on it.
    """
    for label, bbox in bounding_boxes.items():
        # Extract bounding box coordinates
        (min_x, min_y, min_z), (max_x, max_y, max_z) = bbox

        # y
        mask[min_x:max_x + 1, min_y:min_y + 1, min_z:max_z + 1] = label  # Bottom face
        mask[min_x:max_x + 1, max_y:max_y + 1, min_z:max_z + 1] = label  # Top face

        # z
        mask[min_x:max_x + 1, min_y:max_y + 1, min_z:min_z + 1] = label  # Front face
        mask[min_x:max_x + 1, min_y:max_y + 1, max_z:max_z + 1] = label

        # x
        mask[min_x:min_x + 1, min_y:max_y + 1, min_z:max_z + 1] = label  # Left face
        mask[max_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1] = label  # Right face

    return mask


def plot_bounding_box(image, mask_image, num_objects):
    import matplotlib.patches as patches
    # Create a subplot with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the image
    axes[0].imshow(image[:, :, image.shape[2] // 2], cmap='gray')
    axes[0].set_title('NIfTI Image')

    # Plot the mask with bounding boxes
    axes[1].imshow(image[:, :, image.shape[2] // 2], cmap='gray')
    axes[1].set_title('Mask with Bounding Boxes')

    for label in range(1, num_objects + 1):
        # Find coordinates of the bounding box
        nonzero_indices = np.where(mask_image == label)
        min_x, min_y, min_z = np.min(nonzero_indices, axis=1)
        max_x, max_y, max_z = np.max(nonzero_indices, axis=1)

        # Create a rectangle patch
        rect = patches.Rectangle((min_y, min_x), max_y - min_y, max_x - min_x,
                                 linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle patch to the plot
        axes[1].add_patch(rect)

    plt.show()

    pass