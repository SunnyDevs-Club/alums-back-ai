"""
module "dataset.py"
Module to generate Pixel-Set format data for L-TAE from initial row data
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import stackstac
import pystac_client
from shapely.geometry import Polygon

RES_10M_BANDS = ["blue", "green", "red", "nir"]
RES_20M_BANDS = ["rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22"]
