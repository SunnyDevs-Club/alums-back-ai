"""
module "dataset.py"
Module to generate Training Data for L-TAE from initial row data 

RAW DATA:
    Monitoring Report of Andijan and Navoiy Regions 
    from May 2024; retrieved from the Ministry of Agriculture of the Republic of Uzbekistan)
    This approach is expandable to multiple data sources (several reports) if kept same data format.

POLYGONS:
    Store in .geojson format.
    Retrieved from the Ministry of Agriculture of the Republic of Uzbekistan
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import pystac_client
import rioxarray
import shapely
import stackstac
from scipy import stats as st
from tqdm import tqdm

import os
import shutil
import json
import pickle


DATA_FOLDER = './data'
STAC_SERVICE_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = ['sentinel-2-l2a']
PLATFORM = "sentinel-2b"
RES_10M_BANDS = ["blue", "green", "red", "nir"]
RES_20M_BANDS = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]


CROP_DATES = {
    "Paxta": '2024-04-01/2024-09-30',
    "G'alla": '2024-02-01/2024-06-30',
    "Bog'": '2024-03-01/2024-09-30'
}


# Define the mapping from crop types to labels
CROP_LABELS = {
    "Paxta": 0,
    "G'alla": 1,
    "Bog'": 2,
    "Others": 3
}


def _prep_xlsx_to_df(file_path) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    df.rename(columns={
        df.columns[1]: 'viloyat',
        df.columns[2]: 'tuman',
        df.columns[3]: 'massiv',
        df.columns[4]: 'kontur_raqami',
        df.columns[5]: 'kesma_raqami',
        df.columns[6]: 'x',
        df.columns[7]: 'y',
        df.columns[10]: 'crop_type'
    }, inplace=True)

    df.dropna(subset=['x', 'y', 'crop_type'], inplace=True)

    # Drop unnecessary columns
    df.drop(df.columns[[0,1,3,8,9]], inplace=True, axis=1)

    return df


def _get_dfs(dir_path) -> pd.DataFrame:
    df = pd.DataFrame()

    for f in os.scandir(dir_path):
        if f.is_file() and f.name.endswith('.xlsx'):
            temp_df = _prep_xlsx_to_df(f.path)
            df = pd.concat([df, temp_df], ignore_index=True)
    
    return df


def filter_crops(dir_path) -> pd.DataFrame:
    """
    Process the data from a given directory and classify rows with crop types
    ["Bog'", "G'alla", "Paxta"]

    Args:
        dir_path (str): The path to the directory containing the data.

    Returns:
        pd.DataFrame: A DataFrame with filtered and modified rows.

    Raises:
        ValueError: If the provided directory path does not exist.
    """
    if not os.path.exists(dir_path):
        raise ValueError("Path do not exists")

    df = _get_dfs(dir_path)

    crop_types_count = df.value_counts(subset=['crop_type'])
    valid_crop_types = crop_types_count[crop_types_count > 45].index.get_level_values(0)

    df = df[df['crop_type'].isin(valid_crop_types)]

    if 'Shudgor' in valid_crop_types:
        df = df[df['crop_type'] != 'Shudgor']
    
    if 'Yer yong\'oq' in valid_crop_types:
        df = df[df['crop_type'] != 'Yer yong\'oq']
    
    if "Bo'sh yer" in valid_crop_types:
        df = df[df['crop_type'] != 'Bo\'sh yer']
    
    return df


def get_others(dir_path) -> pd.DataFrame:
    """
    Process the data from a given directory and classify rows with crop types
    other than ["Bog'", "G'alla", "Paxta"] as 'Others'. Limit to 120 rows.

    Args:
        dir_path (str): The path to the directory containing the data.

    Returns:
        pd.DataFrame: A DataFrame with filtered and modified rows.

    Raises:
        ValueError: If the provided directory path does not exist.
    """

    if not os.path.exists(dir_path):
        raise ValueError("Path does not exist")

    # Assume _get_dfs is a helper function that reads files and returns a concatenated DataFrame
    df = _get_dfs(dir_path)

    # Filter rows where crop type is not in the specified classes
    excluded_classes = ["Bog'", "G'alla", "Paxta"]
    filtered_df = df[~df['crop_type'].isin(excluded_classes)]

    # Assign crop type as 'Others'
    filtered_df['crop_type'] = 'Others'

    return filtered_df


def _df_to_gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        data=df, geometry=gpd.points_from_xy(df.y, df.x), crs='EPSG:4326'
    )


def _prep_json_to_gdf(path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)

    gdf = gdf[['Shape_Length', 'Shape_Area', 'geometry']]
    gdf['geom_copy'] = gdf['geometry']

    return gdf


def _get_gdfs(dir_path) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame()

    for f in os.scandir(dir_path):
        if f.is_file() and f.name.endswith('.geojson'):
            temp_gdf = _prep_json_to_gdf(f.path)
            gdf = pd.concat([gdf, temp_gdf], ignore_index=True)
    
    return gdf


def link_polygons(dir_path, points_df: pd.DataFrame) -> gpd.GeoDataFrame:
    if not os.path.exists(dir_path):
        raise ValueError("Path do not exists")

    points_gdf = _df_to_gdf(points_df)

    gdf = _get_gdfs(dir_path=dir_path)
    linked_gdf: gpd.GeoDataFrame = gpd.sjoin(points_gdf, gdf, how='left', predicate='within')

    final_gdf: gpd.GeoDataFrame = linked_gdf[linked_gdf['index_right'].notna()]
    final_gdf = final_gdf.drop(columns=['geometry', 'index_right'])
    final_gdf = final_gdf.rename(columns={'geom_copy': 'geometry'})

    final_gdf = final_gdf[final_gdf['geometry'].is_valid]

    final_gdf = final_gdf.drop_duplicates(subset=['kontur_raqami', 'geometry'])
    final_gdf = final_gdf.sort_values(by=['tuman', 'kontur_raqami', 'kesma_raqami'])
    
    return final_gdf


def _create_stack(items, bands, bounds) -> xr.Dataset:
    return stackstac.stack(
        items=items,
        assets=bands,
        gdal_env=stackstac.DEFAULT_GDAL_ENV.updated({
            'GDAL_HTTP_MAX_RETRY': 5,
            'GDAL_HTTP_RETRY_DELAY': 5,
        }),
        epsg=4326,
        chunksize=(1, 1, 50, 50),
        bounds=bounds
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
    ).item_collection()

    sentinel_stack = _create_stack(items=items, bands=RES_10M_BANDS + RES_20M_BANDS, bounds=polygon.bounds)
    sentinel_stack = _delete_duplicates(sentinel_stack)

    try:
        sentinel_stack.load()
    except Exception as e:
        raise ValueError("Error in loading")

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


def main():
    print("Filtering Crops...")
    filtered_crops = get_others(f"{DATA_FOLDER}/raw")
    print("done")

    print("Linking Crops to corresponding polygons...")
    polygons_gdf = link_polygons(f'{DATA_FOLDER}/polygons', filtered_crops)

    crop_counts = polygons_gdf.value_counts(subset=['crop_type'])
    print(crop_counts)
    print('done')

    print("Starting pySTAC Client...")
    client = pystac_client.Client.open(STAC_SERVICE_URL)

    if os.path.exists(f'{DATA_FOLDER}/final'):
        shutil.rmtree(f'{DATA_FOLDER}/final')

    os.mkdir(f'{DATA_FOLDER}/final')
    os.mkdir(f'{DATA_FOLDER}/final/DATA')
    os.mkdir(f'{DATA_FOLDER}/final/META')

    id_ = 0
    geom_features_collection = dict()

    labels_collection = {"label_MoA_protoclass": {}}
    shapes_collection = {}

    # Initialize variables for incremental mean and std
    running_mean = 0
    running_var = 0
    total_pixels = 0

    for _, row in tqdm(polygons_gdf.iterrows(), desc='Time Series data generation', total=len(polygons_gdf)):
        try:
            geom: shapely.Polygon = row.geometry
            if not shapely.is_valid(geom):
                continue
            
            dates_range = CROP_DATES.get(row.crop_type, CROP_DATES['Paxta'])
            arr = generate_time_series(polygon=row.geometry, stac_client=client, collections=COLLECTION, dates_range=dates_range)
            id_ += 1

            # Update means and stds using the helper function
            batch_pixels = arr.shape[2]  # Number of spatial pixels
            batch_mean = np.mean(arr, axis=(0, 2))  # Mean across time and spatial dimensions
            batch_var = np.var(arr, axis=(0, 2))    # Variance across time and spatial dimensions

            running_mean, running_var, total_pixels = update_mean_std(
                running_mean, running_var, total_pixels, batch_mean, batch_var, batch_pixels
            )

            geom_features = generate_geom_features(row.geometry, arr)
            geom_features_collection[str(id_)] = geom_features
            np.save(f'{DATA_FOLDER}/final/DATA/{id_}.npy', arr)

            crop_label = CROP_LABELS[row.crop_type]
            labels_collection["label_MoA_protoclass"][str(id_)] = crop_label

            shapes_collection[str(id_)] = arr.shape[0]
        
        except ValueError:
            continue
        except Exception as e:
            running_std = np.sqrt(running_var)
            with open(f'{DATA_FOLDER}/final/S2-2024-meanstd.pkl', 'wb') as f:
                pickle.dump((running_mean, running_std), f)
            
            # Save geometric features
            with open(f'{DATA_FOLDER}/final/META/geomfeat.json', 'w') as file:
                json.dump(geom_features_collection, file, indent=4, default=convert_numpy)

            with open(f'{DATA_FOLDER}/final/META/shapes.json', 'w') as file:
                json.dump(shapes_collection, file, indent=4, default=convert_numpy)

            # Save labels to labels.json
            with open(f'{DATA_FOLDER}/final/META/labels.json', 'w') as label_file:
                json.dump(labels_collection, label_file, indent=4, default=convert_numpy)
            
            raise e


    # Final standard deviation
    running_std = np.sqrt(running_var)

    # Save normalization values
    with open(f'{DATA_FOLDER}/final/S2B-2024-meanstd.pkl', 'wb') as f:
        pickle.dump((running_mean, running_std), f)

    # Save geometric features
    with open(f'{DATA_FOLDER}/final/META/geomfeat.json', 'w') as file:
        json.dump(geom_features_collection, file, indent=4, default=convert_numpy)

    with open(f'{DATA_FOLDER}/final/META/shapes.json', 'w') as file:
        json.dump(shapes_collection, file, indent=4, default=convert_numpy)

    # Save labels to labels.json
    with open(f'{DATA_FOLDER}/final/META/labels.json', 'w') as label_file:
        json.dump(labels_collection, label_file, indent=4, default=convert_numpy)


if __name__ == '__main__':
    main()
