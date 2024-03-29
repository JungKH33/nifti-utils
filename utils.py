import numpy as np

def filter_array(arr, values_to_keep, replacement_value=0):
    """
    Filter a NumPy array, setting elements not in the list of values_to_keep to target_value.

    Args:
        arr (np.ndarray): Input NumPy array.
        values_to_keep (list): List of values to keep in the filtered array.
        replacement_value (int or float, optional): Value to set for elements not in values_to_keep. Default is 0.

    Returns:
        np.ndarray: Filtered NumPy array.

    Example:
        >>> arr = np.array([[1, 2, 3],
        ...                 [4, 5, 6],
        ...                 [7, 8, 9]])
        >>> values_to_keep = [2, 5, 8]
        >>> target_value = 0
        >>> filtered_arr = filter_array(arr, values_to_keep, target_value)
        [[0 2 0]
         [0 5 0]
         [0 8 0]]
    """
    filtered_arr = np.where(np.isin(arr, values_to_keep), arr, replacement_value)
    return filtered_arr