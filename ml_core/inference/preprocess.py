import os
import shutil
import json
import pickle as pkl
from typing import List, Tuple

import geopandas as gpd
import pandas as pd
import xarray as xr
import rioxarray
import stackstac
import shapely
import pystac_client
import numpy as np
import scipy.stats as st

from dask.diagnostics import ProgressBar
from tqdm import tqdm

from ml_core.config import Config
from ml_core.inference.utils import convert_numpy


polygons = gpd.read_file(Config.PROJ_DIR / 'data' / 'polygons_data.geojson')

def retrieve_rows_by_pairs(gdf: gpd.GeoDataFrame, pairs: List[Tuple[float | int, str]]):
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
    config: Config
) -> np.ndarray:
    items = stac_client.search(
        collections=config.COLLECTION,
        intersects=polygon,
        datetime=config.DATE_RANGE,
        query={
            "platform": "sentinel-2b",
            "eo:cloud_cover": {"lt": 20}
        }
    ).item_collection()

    sentinel_stack = _create_stack(items=items, bands=config.RES_10M_BANDS + config.RES_20M_BANDS, bounds=polygon.bounds)
    sentinel_stack = _delete_duplicates(sentinel_stack)

    with ProgressBar():
        sentinel_stack.load()

    for band in config.RES_10M_BANDS + config.RES_20M_BANDS:
        rescaled_vals = 1000 * (sentinel_stack[band] + 0.1)
        sentinel_stack[band] = rescaled_vals

    # Clip to the parcel's boundaries
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


def prepare_inference_data(data_pairs: list[tuple[float | int, str]], config: Config) -> None:
    print("Starting pySTAC Client...")
    client = pystac_client.Client.open(config.STAC_SERVICE_URL)

    if config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)

    os.mkdir(config.DATASET_DIR)
    os.mkdir(config.DATASET_DIR / 'DATA')
    os.mkdir(config.DATASET_DIR / 'META')

    geom_features_collection = dict()

    labels_collection = {"label_MoA_protoclass": {}}
    shapes_collection = {}

    # Initialize variables for incremental mean and std
    running_mean = 0
    running_var = 0
    total_pixels = 0

    current_polygons = retrieve_rows_by_pairs(polygons, data_pairs)

    for _, row in tqdm(current_polygons.iterrows(), desc='Time Series data generation', total=len(current_polygons)):
        try:
            parcel_id = row.id
            geom: shapely.Polygon = row.geometry
            if not shapely.is_valid(geom):
                continue
            arr = generate_time_series(polygon=row.geometry,
                                       stac_client=client,
                                       config=config
                                       )

            # Update means and stds using the helper function
            batch_pixels = arr.shape[2]  # Number of spatial pixels
            batch_mean = np.mean(arr, axis=(0, 2))  # Mean across time and spatial dimensions
            batch_var = np.var(arr, axis=(0, 2))  # Variance across time and spatial dimensions

            running_mean, running_var, total_pixels = update_mean_std(
                running_mean, running_var, total_pixels, batch_mean, batch_var, batch_pixels
            )

            geom_features = generate_geom_features(row.geometry, arr)
            geom_features_collection[str(parcel_id)] = geom_features
            np.save(config.DATASET_DIR / f'DATA/{parcel_id}.npy', arr)

            crop_label = config.CROP_LABELS[row.crop_type]
            labels_collection["label_MoA_protoclass"][str(parcel_id)] = crop_label

            shapes_collection[str(parcel_id)] = arr.shape[0]
        except ValueError:
            pid = parcel_id if 'parcel_id' in locals() else getattr(row, 'id', '<unknown>')
            print(f"Error generating Time-Series for parcel: {pid}")
            continue

    # Save geometric features
    running_std = np.sqrt(running_var)
    with open(config.DATASET_DIR / 'S2-2024-meanstd.pkl', 'wb') as f:
        pkl.dump((running_mean, running_std), f)

    with open(config.DATASET_DIR / 'META/geomfeat.json', 'w') as file:
        json.dump(geom_features_collection, file, indent=4, default=convert_numpy)

    with open(config.DATASET_DIR / 'META/shapes.json', 'w') as file:
        json.dump(shapes_collection, file, indent=4, default=convert_numpy)

    # Save labels to labels.json
    with open(config.DATASET_DIR / 'META/labels.json', 'w') as label_file:
        json.dump(labels_collection, label_file, indent=4, default=convert_numpy)
