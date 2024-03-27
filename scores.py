import numpy as np
from scipy.spatial.distance import cdist

import experimental


def dice_score(data1: np.ndarray, data2: np.ndarray, labels= None) -> dict:
    """
    Calculate the Dice coefficient for multiple labels between two arrays.

    Args:
        data1 (np.ndarray): The first array containing labels.
        data2 (np.ndarray): The second array containing labels (predicted).
        labels (list, optional): List of labels for which to calculate Dice coefficients.
            If None, all unique labels in data1 and data2 will be considered.

    Returns:
        dict: A dictionary containing Dice coefficients for each label.

    Raises:
        ValueError: If the shapes of the input arrays are not equal.

    Example:
        >>> data1 = np.array([[0, 0, 1, 1, 2]])
        >>> data2 = np.array([[0, 1, 1, 0, 2]])
        >>> dice_score(data1, data2)
        {0: 0.5, 1: 0.5, 2: 1.0}
    """
    if not np.array_equal(data1.shape, data2.shape):
        raise ValueError("The shapes of two arrays must be equal. data1 shape: {}, data2 shape: {}".format(data1.shape, data2.shape))

    if labels is None:
        labels = np.unique(np.concatenate((data1, data2)))

    dice_scores = {}
    for label in labels:
        truth_label = data1 == label
        pred_label = data2 == label
        intersection = np.sum(truth_label * pred_label)
        dice = (2.0 * intersection) / (np.sum(truth_label) + np.sum(pred_label))
        dice_scores[label] = dice

    return dice_scores

def assd_score(data1: np.ndarray, data2: np.ndarray, labels= [1]) -> dict:
    """
    Calculate the Average Symmetric Surface Distance (ASSD) score for multiple labels between two arrays.

    Args:
        data1 (np.ndarray): The first array containing labels.
        data2 (np.ndarray): The second array containing labels (predicted).
        labels (list, optional): List of labels for which to calculate ASSD scores.
            If None, all unique labels in data1 and data2 will be considered.

    Returns:
        dict: A dictionary containing ASSD scores for each label.

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
        {1: 0.35}
    """
    if not np.array_equal(data1.shape, data2.shape):
        raise ValueError("The shapes of two arrays must be equal. data1 shape: {}, data2 shape: {}".format(data1.shape, data2.shape))

    if labels is None:
        labels = np.unique(np.concatenate((data1, data2)))

    assd_scores = {}
    for label in labels:

        label1 = data1 == label
        label2 = data2 == label

        if not np.any(label1) or not np.any(label2):
            assd_scores[label] = np.nan
            continue

        # Convert binary arrays to sets of coordinates
        coordinates1 = np.array(np.where(label1)).T
        coordinates2 = np.array(np.where(label2)).T

        # Calculate pairwise distances between points on the surfaces
        distances_1_to_2 = cdist(coordinates1, coordinates2)
        distances_2_to_1 = cdist(coordinates2, coordinates1)

        # Calculate minimum distances from each point on one surface to the other
        min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
        min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)

        # Average the minimum distances to get the ASSD score
        assd = (np.mean(min_distances_1_to_2) + np.mean(min_distances_2_to_1)) / 2.0
        assd_scores[label] = assd

        ###
        print()
        print(np.argmax(distances_1_to_2, axis=1))
        print(np.argmax(distances_2_to_1, axis=1))
        ###

    return assd_scores


if __name__ == '__main__':
    from data import *
    import os
    #gt_path = r"E:\dataset\seg10\ss_mask_new\irb82_0034.nii.gz"
    #inf_path = r"C:\Users\Neurophet\Downloads\irb82_0034.nii.gz"

    gt_dir = r"E:\dataset\seg10\ss_mask_new"
    inf_dir = r"E:\outputs\ss\inference_pp"
    mask_dir = r"E:\dataset\seg10\mask_robust_merged_reorient"
    dice_scores = []
    assd_scores = []
    overlaps = []
    filenames = []
    for filename in os.listdir(gt_dir):
        if filename == 'irb82_0072.nii.gz' or filename == 'irb82_0042.nii.gz' or filename == 'irb82_0048.nii.gz':
            continue
        gt_path = os.path.join(gt_dir, filename)
        inf_path = os.path.join(inf_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        gt_img = load_nii(gt_path)
        inf_img = load_nii(inf_path)
        mask_img = load_nii(mask_path)

        gt_data = nifti_to_numpy(gt_img)
        inf_data = nifti_to_numpy(inf_img)
        mask_data = nifti_to_numpy(mask_img)

        dice = dice_score(gt_data, inf_data)
        assd = assd_score(gt_data, inf_data)
        overlap = experimental.intersection_score(inf_data, mask_data)

        filenames.append(filename)
        dice_scores.append(dice[1])
        assd_scores.append(assd[1])
        overlaps.append(overlap[1])

        print(filename)
        print("DICE Scores : ", dice[1])
        print("ASSD Score : ", assd[1])
        print("Overlapping region : ", overlap[1])

    average_dice_score = sum(dice_scores) / len(dice_scores)
    average_assd_score = sum(assd_scores) / len(assd_scores)

    import matplotlib.pyplot as plt

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(dice_scores, assd_scores)

    # Adding labels and title
    plt.xlabel('Dice Scores')
    plt.ylabel('ASSD Scores')
    plt.title('ASSD Scores vs Dice Scores')

    # Adding filenames as labels
    for i, filename in enumerate(filenames):
        plt.annotate(filename, (dice_scores[i], assd_scores[i]), textcoords="offset points", xytext=(0, 10),
                     ha='center')

    # Displaying the plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(average_dice_score)
    print(average_assd_score)


