import pickle as pkl

import torch.utils.data as data

from ml_core.data.dataset import PixelSetData
from ml_core.utils import custom_collate_fn

from ml_core.config import Config


def __load_dt(config: Config) -> PixelSetData:
    mean_std = pkl.load(open(config.DATASET_DIR / 'S2-2024-meanstd.pkl', 'rb'))
    return PixelSetData(
        str(config.DATASET_DIR),
        labels='label_MoA_protoclass',
        npixel=config.N_PIXEL,
        norm=mean_std,
        extra_feature='geomfeat' if config.GEOMETRIC_FEATURES else None,
        return_id=True
    )


def load_dl(config: Config) -> data.DataLoader:
    dt = __load_dt(config)
    return data.DataLoader(
        dt,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        collate_fn=custom_collate_fn
    )