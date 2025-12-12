import numpy as np

def labels_to_crops(predictions: np.ndarray, label_mapping):
    """
    Convert numerical labels back to string crop type names.

    Parameters:
    -----------
    predictions : np.ndarray
        Array of shape (N, 3) containing [ids, y_true, y_pred]
        where y_true and y_pred are numerical labels
    label_mapping : dict
        Mapping from numerical labels to crop type strings

    Returns:
    --------
    dict
        Dictionary with keys 'ids', 'y_true', 'y_pred' containing
        the predictions with string labels for y_true and y_pred
    """
    ids = predictions[:, 0].astype(int)
    y_true = np.array([label_mapping.get(int(label), f'Unknown_{label}')
                       for label in predictions[:, 1]])
    y_pred = np.array([label_mapping.get(int(label), f'Unknown_{label}')
                       for label in predictions[:, 2]])

    return {
        'ids': ids,
        'y_true': y_true,
        'y_pred': y_pred
    }
