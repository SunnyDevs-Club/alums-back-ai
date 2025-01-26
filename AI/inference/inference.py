import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
import pandas as pd
import pprint
import argparse

import pickle as pkl

from dataset import PixelSetData_inference
from models.stclassifier import PseTae_pretrained


def prepare_model_and_loader(config):
    """
    Prepare the model and data loader for inference.

    Args:
        config (dict): Configuration dictionary with required parameters.

    Returns:
        tuple: Model and data loader objects.
    """
    mean_std = pkl.load(open(config['dataset_folder'] + '/S2B-2024-meanstd.pkl', 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None

    # Configure PixelSetInferenceData for inference
    dt = PixelSetData_inference(
        folder=config['dataset_folder'],
        norm=mean_std,
        npixel=config['npixel'],
        extra_feature=extra,
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

    # Load the pretrained model
    model = PseTae_pretrained(
        config['weight_dir'], model_config,
        device=config['device'], fold=config['fold']
    )

    return model, dl


def recursive_todevice(x, device):
    """
    Recursively move data to the specified device.

    Args:
        x: Input data (tensor, list, etc.).
        device: Target device (e.g., 'cuda' or 'cpu').

    Returns:
        Same structure as x with data moved to the target device.
    """
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def generate_predictions(model, loader, config):
    """
    Generate predictions using the model and store results in a pandas DataFrame.

    Args:
        model: The trained model.
        loader: Data loader for input data.
        config: Configuration dictionary with required parameters.

    Returns:
        pd.DataFrame: DataFrame containing inference results.
    """
    records = []  # Collect prediction data
    device = torch.device(config['device'])

    for (x, ids) in tqdm(loader):
        ids = list(ids)

        x = recursive_todevice(x, device)
        with torch.no_grad():
            prediction = model(x)
        y_pred = list(prediction.argmax(dim=1).cpu().numpy())

        # Combine results
        for id_, pred in zip(ids, y_pred):
            records.append({"id": id_, "predicted_label": pred})

    # Convert records to a pandas DataFrame
    results_df = pd.DataFrame(records)
    return results_df


def run_inference(config):
    """
    Backend interface to prepare the model, load data, and generate predictions for all parcels.

    Args:
        config (dict): Configuration dictionary with required parameters.

    Returns:
        pd.DataFrame: DataFrame containing inference results.
    """
    model, loader = prepare_model_and_loader(config)

    # Generate predictions and get results as a DataFrame
    results_df = generate_predictions(model, loader, config)

    # Optionally save to CSV or process further
    results_df.to_csv("inference_results.csv", index=False)
    return results_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--fold', default='', type=str,
                        help='Path to the Fold folder where the pre-trained weights are saved.')
    parser.add_argument("--weight_dir", default='./results')
    parser.add_argument('--res_dir', default='./results', help='Path to the folder where the results should be stored')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loading workers')
    parser.add_argument('--rdm_seed', default=1, type=int, help='Random seed')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu)')
    parser.add_argument('--display_step', default=50, type=int,
                        help='Interval in batches between display of training metrics')
    parser.add_argument('--preload', dest='preload', action='store_true',
                        help='If specified, the whole dataset is loaded to RAM at initialization')
    parser.set_defaults(preload=False)

    # Training parameters
    parser.add_argument('--kfold', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs per fold')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--gamma', default=1, type=float, help='Gamma parameter of the focal loss')
    parser.add_argument('--npixel', default=64, type=int, help='Number of pixels to sample from the input images')

    # Architecture Hyperparameters
    ## PSE
    parser.add_argument('--input_dim', default=10, type=int, help='Number of channels of input images')
    parser.add_argument('--mlp1', default='[10,32,64]', type=str, help='Number of neurons in the layers of MLP1')
    parser.add_argument('--pooling', default='mean_std', type=str, help='Pixel-embeddings pooling strategy')
    parser.add_argument('--mlp2', default='[132,128]', type=str, help='Number of neurons in the layers of MLP2')
    parser.add_argument('--geomfeat', default=1, type=int,
                        help='If 1 the precomputed geometrical features (f) are used in the PSE.')

    ## TAE
    parser.add_argument('--n_head', default=4, type=int, help='Number of attention heads')
    parser.add_argument('--d_k', default=32, type=int, help='Dimension of the key and query vectors')
    parser.add_argument('--mlp3', default='[512,128,128]', type=str, help='Number of neurons in the layers of MLP3')
    parser.add_argument('--T', default=1000, type=int, help='Maximum period for the positional encoding')
    parser.add_argument('--positions', default='bespoke', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=None, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=20, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 20]', type=str, help='Number of neurons in the layers of MLP4')

    # Command:
    """
    python psetae/finetune.py --dataset_folder ./data/final/ --fold ./pretrained/Fold_3 --num_workers 4 --device cpu --epochs 50 --batch_size 32 --lr 0.0001 --gamma 2 --positions order --lms 100 --dropout 0.3 --mlp4 [128,64,32,5]
    """

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    run_inference(config)