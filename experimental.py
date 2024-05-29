import numpy as np
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

def intersection_score(data1: np.ndarray, data2: np.ndarray, labels=None) -> dict:

    if not np.array_equal(data1.shape, data2.shape):
        raise ValueError(
            "The shapes of two arrays must be equal. data1 shape: {}, data2 shape: {}".format(data1.shape,
                                                                                              data2.shape))

    if labels is None:
        labels = np.unique(np.concatenate((data1, data2)))

    dice_scores = {}

    label = 1
    truth_label = data1 == label
    pred_label = data2 == label
    intersection = np.sum(truth_label * pred_label)
    dice = intersection / np.sum(truth_label)
    dice_scores[label] = dice

    return dice_scores


def merge_masks(data1: np.ndarray, data2: np.ndarray, label1: int = 1, label2: int = 2, new_label: int = 3):
    # Assign labels to data1 and data2
    labeled_data1 = np.where(data1 != 0, label1, 0)
    labeled_data2 = np.where(data2 != 0, label2, 0)

    merged_data = np.zeros_like(data1)  # Initialize merged data with zeros
    overlapping_indices = np.logical_and(labeled_data1 != 0, labeled_data2 != 0)  # Find overlapping indices

    # Copy non-overlapping regions from labeled_data1 and labeled_data2
    merged_data[labeled_data1 != 0] = labeled_data1[labeled_data1 != 0]
    merged_data[labeled_data2 != 0] = labeled_data2[labeled_data2 != 0]

    # Assign new label to overlapping regions
    merged_data[overlapping_indices] = new_label

    return merged_data

def lesion_wise_sensitivity(ground_truth: np.ndarray, predicted: np.ndarray):
    gt_labels = np.unique(ground_truth)
    pred_labels = np.unique(predicted)
    iou_dict = {}

    for gt_label in gt_labels:
        gt_label_mask = (ground_truth == gt_label)
        max_iou = 0
        for pred_label in pred_labels:
            pred_label_mask = (predicted == pred_label)

            # Calculate intersection and union
            intersection = np.logical_and(gt_label_mask, pred_label_mask)
            union = np.logical_or(gt_label_mask, pred_label_mask)

            # Compute IoU
            if np.sum(union) == 0:
                iou = 0.0  # Handle case when both masks are empty
            else:
                iou = np.sum(intersection) / np.sum(union)
                if iou > max_iou:
                    max_iou = iou

        iou_dict[gt_label] = max_iou
    return iou_dict



if __name__ == "__main__":
    from data import *
    from augmentation import *
    import torchio as tio

    input_file = r"E:\dataset\seg10\ss_input_reorient\irb82_0037.nii.gz"
    ss_file = r"E:\dataset\seg10\ss_mask_new\irb82_0037.nii.gz"
    cortex_file = r"E:\dataset\seg10\mask_robust_merged_reorient\irb82_0037.nii.gz"

    img = nifti_to_numpy(load_nii(input_file))
    ss_img = nifti_to_numpy(load_nii(ss_file))
    cortex_img = nifti_to_numpy(reorient(load_nii(cortex_file)))

    print(img.shape)
    print(ss_img.shape)
    print(cortex_img.shape)

    merged = merge_masks(ss_img, cortex_img)

    save_nii('test.nii', merged)
