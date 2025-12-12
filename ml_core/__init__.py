"""
Main Entry Point for backend calls
"""
from typing import Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from ml_core.config import Config
from ml_core.data import load_dl
from ml_core.models import PseTae_pretrained, load_model
from ml_core.processing.preprocess import prepare_inference_data
from ml_core.processing.postprocess import labels_to_crops
from ml_core.utils import recursive_todevice

config: Config = Config()
model: PseTae_pretrained = load_model(config)

def __predict(loader):
    """
    Generate predictions using the model.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    loader : torch.utils.data.DataLoader
        Data loader for processing
    config : dict
        Configuration dictionary

    Returns:
    --------
    np.ndarray
        Array of shape (N, 3) containing [ids, y_true, y_pred]
    """
    record = []
    device = torch.device(config.DEVICE)

    for (x, y, ids) in tqdm(loader):
        y_true = (list(map(int, y)))
        ids = list(ids)

        x = recursive_todevice(x, device)
        with torch.no_grad():
            prediction = model(x)
        y_p = list(prediction.argmax(dim=1).cpu().numpy())

        record.append(np.stack([ids, y_true, y_p], axis=1))

    record = np.concatenate(record, axis=0)

    return record


def predict(pairs: List[Tuple[float | int, str]]):
    # try:
    #     prepare_inference_data(pairs, config)
    # except Exception as e:
    #     print("Error occurred during preprocessing: {}".format(e))
    #     return None

    dl: DataLoader = load_dl(config)
    preds: np.ndarray = __predict(dl)

    results: dict[str, np.ndarray] = labels_to_crops(preds, label_mapping=config.LABEL_TO_CROP)
    return results
