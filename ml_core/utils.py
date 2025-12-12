import torch
import numpy as np


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def convert_numpy(obj):
    if isinstance(obj, np.integer):  # For int64 or other NumPy integer types
        return int(obj)
    elif isinstance(obj, np.floating):  # For NumPy float types
        return float(obj)
    elif isinstance(obj, np.ndarray):  # For NumPy arrays
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def pad_with_value(sequences, value=1):
    """
    Pads a list of sequences with a specified value.

    Args:
        sequences: List of tensors with varying lengths.
        value: The padding value.

    Returns:
        A single tensor with all sequences padded to the same length.
    """
    max_length = max(seq.size(0) for seq in sequences)  # Find the maximum length
    padded = torch.full((len(sequences), max_length, *sequences[0].shape[1:]), value, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        padded[i, :seq.size(0)] = seq  # Copy the original data into the padded tensor

    return padded


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length Pixel-Set sequences and Extra-Features.

    Args:
        batch: A list of samples, where each sample can be:
               - ((Pixel-Set, Pixel-Mask), Label)
               - (((Pixel-Set, Pixel-Mask), Extra-Features), Label)
               - With or without ID.

    Returns:
        Collated batch with padded Pixel-Set sequences and Extra-Features.
    """
    data, labels, ids = [], [], []

    for sample in batch:
        if len(sample) == 3:  # Case with ID
            data.append(sample[0])
            labels.append(sample[1])
            ids.append(sample[2])
        elif len(sample) == 2:  # Case without ID
            data.append(sample[0])
            labels.append(sample[1])

    # Handle nested data structures
    if isinstance(data[0], tuple):  # If data is (Pixel-Set, Pixel-Mask)
        if isinstance(data[0][0], tuple):  # If data is ((Pixel-Set, Pixel-Mask), Extra-Features)
            pixel_set, pixel_mask, extra_features = zip(*[
                (d[0][0], d[0][1], d[1]) for d in data
            ])
            pixel_set = pad_with_value(pixel_set, value=1)  # Pad Pixel-Set with ones
            pixel_mask = pad_with_value(pixel_mask, value=1)  # Pad Pixel-Mask with ones
            extra_features = pad_with_value(extra_features, value=1)  # Pad Extra-Features with ones
            collated_data = ((pixel_set, pixel_mask), extra_features)
        else:  # If data is (Pixel-Set, Pixel-Mask)
            pixel_set, pixel_mask = zip(*data)
            pixel_set = pad_with_value(pixel_set, value=1)  # Pad Pixel-Set with ones
            pixel_mask = pad_with_value(pixel_mask, value=1)  # Pad Pixel-Mask with ones
            collated_data = (pixel_set, pixel_mask)
    else:
        raise ValueError("Unexpected data structure in batch.")

    # Stack labels
    labels = torch.stack(labels)

    # Return results
    if ids:
        return collated_data, labels, ids
    else:
        return collated_data, labels
