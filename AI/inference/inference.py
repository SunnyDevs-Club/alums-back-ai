import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm

import pickle as pkl

from psetae.dataset import PixelSetData
from psetae.models.stclassifier import PseTae_pretrained


def prepare_model_and_loader(config):
    mean_std = pkl.load(open(config['dataset_folder'] + '/S2B-2024-meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    # Configure PixelSetData
    dt = PixelSetData(
        config['dataset_folder'],
        labels=config['label_class'],
        npixel=config['npixel'],
        norm=mean_std,
        extra_feature=extra,
        return_id=True
    )

    dl = data.DataLoader(
        dt,
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    model_config = dict(
        input_dim=config['input_dim'],
        mlp1=config['mlp1'],
        pooling=config['pooling'],
        mlp2=config['mlp2'],
        n_head=config['n_head'],
        d_k=config['d_k'],
        mlp3=config['mlp3'],
        dropout=config['dropout'],
        T=config['T'],
        len_max_seq=config['lms'],
        positions=None,
        mlp4=config['mlp4']
    )

    if config['geomfeat']:
        model_config.update(with_extra=True, extra_size=4)
    else:
        model_config.update(with_extra=False, extra_size=None)

    model = PseTae_pretrained(
        config['weight_dir'], model_config,
        device=config['device'], fold=config['fold']
    )

    return model, dl


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def generate_predictions(model, loader, config):
    record = []
    device = torch.device(config['device'])

    for (x, y, ids) in tqdm(loader):
        y_true = list(map(int, y))
        ids = list(ids)

        x = recursive_todevice(x, device)
        with torch.no_grad():
            prediction = model(x)
        y_p = list(prediction.argmax(dim=1).cpu().numpy())

        record.append(np.stack([ids, y_true, y_p], axis=1))

    record = np.concatenate(record, axis=0)


def run_inference(config):
    """
    Backend interface to prepare the model, load data, and generate predictions for all parcels.

    Args:
        config (dict): Configuration dictionary with required parameters.
    """
    model, loader = prepare_model_and_loader(config)

    generate_predictions(model, loader, config)
