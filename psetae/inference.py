import torch
import torch.utils.data as data
import numpy as np

import stackstac
import shapely
import pystac_client
import rioxarray

import xarray as xr
import pandas as pd
import geopandas as gpd

from scipy import stats as st
from tqdm import tqdm
from dask.diagnostics import ProgressBar

import os
import pickle as pkl
import argparse
import pprint
import json
import shutil


from models.stclassifier import PseTae_pretrained, PseTae_finetune
from dataset import PixelSetData


#########################################################################################################
############                                DATA PREPROCESSING                               ############
#########################################################################################################

DATA_FOLDER = r'C:\Users\m.maxsudov\iut\ALUMS\data'
STAC_SERVICE_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = ['sentinel-2-l2a']
PLATFORM = "sentinel-2b"
RES_10M_BANDS = ["blue", "green", "red", "nir"]
RES_20M_BANDS = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]

DATE_RANGE = '2024-02-01/2024-09-30'

# Define the mapping from crop types to labels
CROP_LABELS = {
    "Paxta": 0,
    "G'alla": 1,
    "Bog'": 2,
}

# Reverse mapping from labels to crop types
LABEL_TO_CROP = {v: k for k, v in CROP_LABELS.items()}


def retrieve_rows_by_pairs(gdf, pairs):
    """
    Retrieve rows from GeoDataFrame given a list of (kontur_raqami, kesma_raqami) pairs.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        The geodataframe to retrieve rows from
    pairs : list of tuples
        List of (kontur_raqami, kesma_raqami) tuples
        Example: [(1.0, 'A'), (2.0, 'B'), (3.0, 'C')]
    
    Returns:
    --------
    GeoDataFrame
        Subset of gdf matching the provided pairs
    """
    if not pairs:
        return gdf.iloc[0:0]  # Return empty GeoDataFrame with same structure
    
    # Create a mask for matching pairs
    mask = False
    for kontur, kesma in pairs:
        pair_mask = (gdf['kontur_raqami'] == kontur) & (gdf['kesma_raqami'] == kesma)
        mask = mask | pair_mask
    
    return gdf[mask]


def _create_stack(items, bands, bounds) -> xr.Dataset:
    return stackstac.stack(
        items=items,
        assets=bands,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated({
            'GDAL_HTTP_MAX_RETRY': 5,
            'GDAL_HTTP_RETRY_DELAY': 5,
            'GDAL_NUM_THREADS': 'ALL_CPUS', 
            'VSI_CACHE': 'TRUE',
        }),
        epsg=4326,
        chunksize=512,
        bounds=bounds,

    ).to_dataset(dim='band')


def _delete_duplicates(stack: xr.Dataset) -> xr.Dataset:
    time_index = pd.to_datetime(stack['time'].values)
    cloud_cover = stack['eo:cloud_cover'].values

    data_info = pd.DataFrame({
        'time': time_index,
        'date': time_index.date,
        'cloud_cover': cloud_cover
    })

    min_cloud_cover_indices = data_info.groupby('date')['cloud_cover'].idxmin()
    return stack.isel(time=min_cloud_cover_indices)


def generate_time_series(
    polygon: shapely.geometry.Polygon,
    stac_client: pystac_client.Client,
    collections: list,
    dates_range: str
) -> np.ndarray:
    items = stac_client.search(
        collections=collections,
        intersects=polygon,
        datetime=dates_range,
        query={
            "platform": "sentinel-2b",
            "eo:cloud_cover": {"lt": 20}
        }
        # max_items=4
    ).item_collection()

    sentinel_stack = _create_stack(items=items, bands=RES_10M_BANDS + RES_20M_BANDS, bounds=polygon.bounds)
    sentinel_stack = _delete_duplicates(sentinel_stack)

    with ProgressBar():
        sentinel_stack.load()

    for band in RES_10M_BANDS + RES_20M_BANDS:
        rescaled_vals = 1000 * (sentinel_stack[band] + 0.1)
        sentinel_stack[band] = rescaled_vals

    # Clip to the percel's boundaries
    sentinel_stack.rio.write_crs('EPSG:4326', inplace=True)
    sentinel_stack = sentinel_stack.rio.clip([polygon], drop=True)

    # Convert dataset to numpy.ndarray
    parcel_arr = sentinel_stack.to_array(dim='band')
    reshaped_arr = parcel_arr.stack(S=('y', 'x'))
    np_arr = reshaped_arr.values
    np_arr = np_arr.transpose((1, 0, 2))
    np_arr[np.isnan(np_arr)] = 0

    return np_arr


def generate_geom_features(polygon: shapely.Polygon, array: np.ndarray) -> list:
    f_perimeter = polygon.length

    non_zero_counts = [
        np.count_nonzero(array[t][c])
        for t in range(array.shape[0])
        for c in range(array.shape[1])
    ]
    f_pixel_count, _ = st.mode(non_zero_counts)
    f_cover_ratio = f_pixel_count / array.shape[2]
    f_perimeter_surface_ratio = f_perimeter / polygon.area

    return [f_perimeter, f_pixel_count, f_cover_ratio, f_perimeter_surface_ratio]


def update_mean_std(running_mean, running_var, total_pixels, batch_mean, batch_var, batch_pixels):
    """
    Incrementally update mean and standard deviation.

    Parameters:
    - running_mean: np.ndarray, current running mean.
    - running_var: np.ndarray, current running variance.
    - total_pixels: int, total pixels processed so far.
    - batch_mean: np.ndarray, mean of the current batch.
    - batch_var: np.ndarray, variance of the current batch.
    - batch_pixels: int, number of pixels in the current batch.

    Returns:
    - updated_mean: np.ndarray, updated mean.
    - updated_var: np.ndarray, updated variance.
    - updated_total_pixels: int, updated total pixel count.
    """
    total_pixels_new = total_pixels + batch_pixels
    delta = batch_mean - running_mean
    updated_mean = running_mean + (batch_pixels / total_pixels_new) * delta
    updated_var = running_var + batch_var + (total_pixels * batch_pixels / total_pixels_new) * (delta ** 2)
    return updated_mean, updated_var, total_pixels_new


def convert_numpy(obj):
    if isinstance(obj, np.integer):  # For int64 or other NumPy integer types
        return int(obj)
    elif isinstance(obj, np.floating):  # For NumPy float types
        return float(obj)
    elif isinstance(obj, np.ndarray):  # For NumPy arrays
        return obj.tolist()
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def prepare_inference_data(data_pairs: list[tuple[float | int, str]], data_folder=DATA_FOLDER, dates_range=DATE_RANGE):
    print("Starting pySTAC Client...")
    client = pystac_client.Client.open(STAC_SERVICE_URL)
    
    if os.path.exists(f'{DATA_FOLDER}/inference'):
        shutil.rmtree(f'{DATA_FOLDER}/inference')

    os.mkdir(f'{DATA_FOLDER}/inference')
    os.mkdir(f'{DATA_FOLDER}/inference/DATA')
    os.mkdir(f'{DATA_FOLDER}/inference/META')

    geom_features_collection = dict()

    labels_collection = {"label_MoA_protoclass": {}}
    shapes_collection = {}

    # Initialize variables for incremental mean and std
    running_mean = 0
    running_var = 0
    total_pixels = 0

    polygons_gdf = gpd.read_file(os.path.join(data_folder, 'polygons_data.geojson'))
    polygons_gdf = retrieve_rows_by_pairs(polygons_gdf, data_pairs)

    for _, row in tqdm(polygons_gdf.iterrows(), desc='Time Series data generation', total=len(polygons_gdf)):
        try:
            parcel_id = row.id
            geom: shapely.Polygon = row.geometry
            if not shapely.is_valid(geom):
                continue
            arr = generate_time_series(polygon=row.geometry, stac_client=client, collections=COLLECTION, dates_range=dates_range)

            # Update means and stds using the helper function
            batch_pixels = arr.shape[2]  # Number of spatial pixels
            batch_mean = np.mean(arr, axis=(0, 2))  # Mean across time and spatial dimensions
            batch_var = np.var(arr, axis=(0, 2))    # Variance across time and spatial dimensions

            running_mean, running_var, total_pixels = update_mean_std(
                running_mean, running_var, total_pixels, batch_mean, batch_var, batch_pixels
            )

            geom_features = generate_geom_features(row.geometry, arr)
            geom_features_collection[str(parcel_id)] = geom_features
            np.save(f'{DATA_FOLDER}/inference/DATA/{parcel_id}.npy', arr)

            crop_label = CROP_LABELS[row.crop_type]
            labels_collection["label_MoA_protoclass"][str(parcel_id)] = crop_label

            shapes_collection[str(parcel_id)] = arr.shape[0]
        except ValueError:
            pid = parcel_id if 'parcel_id' in locals() else getattr(row, 'id', '<unknown>')
            print(f"Error generating Time-Series for parcel: {pid}")
            continue

    # Save geometric features
    running_std = np.sqrt(running_var)
    with open(f'{DATA_FOLDER}/inference/S2-2024-meanstd.pkl', 'wb') as f:
        pkl.dump((running_mean, running_std), f)

    with open(f'{DATA_FOLDER}/inference/META/geomfeat.json', 'w') as file:
        json.dump(geom_features_collection, file, indent=4, default=convert_numpy)

    with open(f'{DATA_FOLDER}/inference/META/shapes.json', 'w') as file:
        json.dump(shapes_collection, file, indent=4, default=convert_numpy)

    # Save labels to labels.json
    with open(f'{DATA_FOLDER}/inference/META/labels.json', 'w') as label_file:
        json.dump(labels_collection, label_file, indent=4, default=convert_numpy)


#########################################################################################################
############                                     INFERENCE                                   ############
#########################################################################################################
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


def prepare_model_and_loader(config):
    mean_std = pkl.load(open(os.path.join(config['dataset_folder'], 'S2-2024-meanstd.pkl'), 'rb'))
    extra = 'geomfeat' if config['geomfeat'] else None
    dt = PixelSetData(config['dataset_folder'], labels='label_MoA_protoclass', npixel=config['npixel'],
                      norm=mean_std,
                      extra_feature=extra, return_id=True)
    dl = data.DataLoader(dt, batch_size=config['batch_size'], num_workers=config['num_workers'], collate_fn=custom_collate_fn)

    model_config = dict(input_dim=config['input_dim'], mlp1=config['mlp1'], pooling=config['pooling'],
                        mlp2=config['mlp2'], n_head=config['n_head'], d_k=config['d_k'], mlp3=config['mlp3'],
                        dropout=config['dropout'], T=config['T'], len_max_seq=config['lms'],
                        positions=dt.date_positions if config['positions'] == 'bespoke' else None,
                        mlp4=config['mlp4'])

    if config['geomfeat']:
        model_config.update(with_extra=True, extra_size=4)
    else:
        model_config.update(with_extra=False, extra_size=None)

    model = PseTae_pretrained(config['weight_dir'], model_config, device=config['device'], fold=config['fold'])
    # model = PseTae_finetune(fold_folder=os.path.join(config['weight_dir'], config['fold']),
    #                          device=config['device'], hyperparameters=model_config)

    return model, dl


def recursive_todevice(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return [recursive_todevice(c, device) for c in x]


def labels_to_crops(predictions, label_mapping=LABEL_TO_CROP):
    """
    Convert numerical labels back to string crop type names.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Array of shape (N, 3) containing [ids, y_true, y_pred]
        where y_true and y_pred are numerical labels
    label_mapping : dict
        Mapping from numerical labels to crop type strings
        Default: LABEL_TO_CROP
    
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


def predict(model, loader, config):
    """
    Generate predictions using the model.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained model
    loader : torch.utils.data.DataLoader
        Data loader for inference
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    np.ndarray
        Array of shape (N, 3) containing [ids, y_true, y_pred]
    """
    record = []
    device = torch.device(config['device'])

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


def main(config):
    print("Preprocessing . . . ")
    prepare_inference_data(data_pairs=[(1232.0, '1285'), (2.0, '2')])
    config['dataset_folder'] = f'{DATA_FOLDER}/inference'
    config['weight_dir'] = './results'

    print('Preparation . . . ')
    model, loader = prepare_model_and_loader(config)
    print('Inference . . .')
    predictions = predict(model, loader, config)
    results = labels_to_crops(predictions)

    # Optionally save results if output_dir is specified
    if config.get('output_dir'):
        os.makedirs(config['output_dir'], exist_ok=True)
        np.save(os.path.join(config['output_dir'], 'Predictions_id_ytrue_y_pred.npy'), predictions)
        print('Results stored in directory {}'.format(config['output_dir']))
    
    print(results)
    
    return results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Set-up parameters
    parser.add_argument('--dataset_folder', default='', type=str,
                        help='Path to the folder where the results are saved.')
    parser.add_argument('--weight_dir', default='', type=str,
                        help='Path to the folder containing the model weights')
    parser.add_argument('--fold', default='1', type=str,
                        help='Specify whether to load the weight sets of all folds (all) or '
                             'only load the weight of a specific fold by indicating its number')
    parser.add_argument('--output_dir', default='./output',
                        help='Path to the folder where the predictions should be stored')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers')
    parser.add_argument('--device', default='cpu', type=str,
                        help='Name of device to use for tensor computations (cuda/cpu/xpu)')

    # Dataset parameters
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
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
    parser.add_argument('--positions', default='order', type=str,
                        help='Positions to use for the positional encoding (bespoke / order)')
    parser.add_argument('--lms', default=100, type=int,
                        help='Maximum sequence length for positional encoding (only necessary if positions == order)')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout probability')

    ## Classifier
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--mlp4', default='[128, 64, 32, 3]', type=str, help='Number of neurons in the layers of MLP4')

    config = parser.parse_args()
    config = vars(config)
    for k, v in config.items():
        if 'mlp' in k:
            v = v.replace('[', '')
            v = v.replace(']', '')
            config[k] = list(map(int, v.split(',')))

    pprint.pprint(config)
    main(config)
