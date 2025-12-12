import numpy as np


def convert_numpy(obj):
    if isinstance(obj, np.integer):  # For int64 or other NumPy integer types
        return int(obj)
    elif isinstance(obj, np.floating):  # For NumPy float types
        return float(obj)
    elif isinstance(obj, np.ndarray):  # For NumPy arrays
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")
