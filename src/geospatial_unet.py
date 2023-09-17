# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # **Geos.pathatial Analysis with Machine Learning**
#

# %% [markdown]
# # Imports & Environment Settings
#

# %% [markdown]
# ## Imports
#
# <!-- _You may see a message about regarding the use of a Tensorflow binary that is optimized with oneAPI Deep Neural Network Library (oneDNN). There is nothing wrong and it can be safely ignored._ -->
#
# Run the following cell to install and import neccessary libraries for this workflow. A temporary directory is created to store the data and model files. 
#
# > Please note that if you rerun this cell, the temporary directory will be deleted and recreated. If you want to keep the data and model files, please copy them to a permanent location before rerunning this cell.
#

# %%
# import libraries & create a temporary working directory in current folder

# # !pip install --quiet contextily earthpy fiona geopandas rasterio pyproj keras-spatial spectral

import itertools
import os
import pathlib
import random
import re
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import alphashape
import contextily as cx
import cv2
import earthpy.spatial as es
import fiona as fio
import geopandas as gpd
import imgaug as ia
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import spectral
import tensorflow as tf
# import tensorflow_addons as tfa
import torch
from imgaug import augmenters as iaa
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras_spatial import SpatialDataGenerator
from osgeo import gdal
from rasterio import windows
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# %%
START_TIME = time.time()

# load functions and set global variables
warnings.filterwarnings("ignore")

# data dir for temporary/working files in the current directory
working_dir = os.path.join(".", "working")
os.makedirs(working_dir, exist_ok=True)
print(
    f"{time.ctime()}: Created a temporary working directory in current folder at {working_dir}"
)

tf.get_logger().setLevel("INFO")

# helper functions & global variables for the workflow
PIXEL_SIZE = -1
TILE_SIZE = -1


def reproject_raster(in_path, out_path, to_crs):
    # reproject raster to project crs
    with rio.open(in_path) as src:
        if src.crs == to_crs:
            print(f"{time.ctime()}: {in_path} is already in target CRS.")
            return in_path

        src_crs = src.crs
        transform, width, height = calculate_default_transform(
            src_crs, to_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()

        kwargs.update(
            {"crs": to_crs, "transform": transform, "width": width, "height": height}
        )

        with rio.open(out_path, "w", **kwargs) as dst:
            for i in tqdm(range(1, src.count + 1)):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=to_crs,
                    resampling=Resampling.nearest,
                )

    return out_path


def encode(data_path, feature):
    # create a numeric unique value for each attribute/feature in the data feature
    # vector = data.copy()

    data = gpd.read_file(data_path)
    data.columns = map(str.lower, data.columns)

    data = data.dropna(subset=["geometry"])

    feature = feature.lower()

    le = preprocessing.LabelEncoder()
    le.fit(data[feature])

    data[f"{feature}_encoded"] = le.transform(data[feature])
    data[f"{feature}_encoded"] += 1

    data.to_file(data_path)


def get_windows(window_shape, image_shape):
    win_rows, win_cols = window_shape
    img_rows, img_cols = image_shape
    offsets = itertools.product(range(0, img_cols, win_cols), range(0, img_rows, win_rows))
    image_window = windows.Window(col_off=0, row_off=0, width=img_cols, height=img_rows)

    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=win_cols, height=win_rows
        )

        yield window.intersection(image_window)


def to_raster(
    data_path,
    output_path,
    feature_id,
    pixel_size=PIXEL_SIZE,
    dtype="float64",
    windows_shape=(1024, 1024),
):
    encode(data_path, feature_id)

    with fio.open(data_path) as features:
        crs = features.crs
        xmin, ymin, xmax, ymax = features.bounds
        transform = rio.Affine.from_gdal(xmin, pixel_size, 0, ymax, 0, -pixel_size)
        out_shape = (int((ymax - ymin) / pixel_size), int((xmax - xmin) / pixel_size))

        with rio.open(
            output_path,
            "w",
            height=out_shape[0],
            width=out_shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            tiled=True,
            options=["COMPRESS=LZW"],
        ) as raster:
            for window in get_windows(windows_shape, out_shape):
                window_transform = windows.transform(window, transform)
                # can be smaller than windows_shape at the edges
                window_shape = (window.height, window.width)
                window_data = np.zeros(window_shape)

                for feature in features:
                    value = feature["properties"][f"{feature_id}_encoded"]
                    geom = feature["geometry"]
                    d = rasterize(
                        [(geom, value)],
                        all_touched=False,
                        out_shape=window_shape,
                        transform=window_transform,
                    )
                    window_data += d  # sum values up

                raster.write(window_data, window=window, indexes=1)


# function to find the number closest to n and divisible by m
def closestDivisibleNumber(n, m):
    if n % m == 0:
        return n

    return n - (n % m)


def crop_center(img, cropx, cropy):
    threeD = False
    try:
        _, x, y = img.shape
        threeD = True
    except:
        x, y = img.shape

    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2

    if threeD:
        return img[:, startx : startx + cropx, starty : starty + cropy]

    return img[startx : startx + cropx, starty : starty + cropy]


def concave_hull(dataframe):
    """Create a single concave hull of an input GeoPandas DataFrame"""
    flat_list = []

    # Iterate over each geometry in the DataFrame
    for geom in dataframe["geometry"]:
        # Check if the geometry is a MultiPolygon
        if geom.geom_type == "MultiPolygon":
            # Iterate over each polygon within the MultiPolygon
            for polygon in geom.geoms:
                # Extract the exterior coordinates of the polygon
                flat_list.extend(list(polygon.exterior.coords))
        else:
            # Extract the exterior coordinates of the geometry
            flat_list.extend(list(geom.exterior.coords))

    # Create the concave hull
    vertices = [(x, y) for x, y in flat_list]
    # alpha = alphashape.optimizealpha(vertices) / 2
    hull = alphashape.alphashape(vertices, 0.001)

    # Create a GeoDataFrame with the concave hull
    result = gpd.GeoDataFrame(geometry=[hull], crs=dataframe.crs)

    return result


# %% [markdown]
# ## Setting a random seed

# %% [markdown]
# ### What is a random seed?
#
# A random seed is a number that is used to initialize a pseudorandom number generator. This is used to generate a sequence of numbers that are seemingly random, but are actually deterministic. This is useful for reproducibility, as the same seed will always generate the same sequence of "random" numbers.
#
# In short, this allows the results of this workflow to be reproducible.
#
# Set the random seed to any integer value.
#

# %%
# define a random seed for reproducibility
seed_picker = widgets.IntText(value=42, description="Seed:", disabled=False)

seed_picker

# %%
# set seed
SEED = seed_picker.value

# set `PYTHONHASHSEED` environment variable at a fixed value
os.environ["PYTHONHASHSEED"] = str(SEED)

# set `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)

# set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)

# set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(SEED)

# configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

torch.manual_seed(SEED)

# %% [markdown]
# # Add Data

# %% [markdown]
# ## Upload your data
#

# %% [markdown]
# ### Features
#
# Features are the inputs the model learns in order to predict a _mask_. For example, if you want to predict the land cover of a region, a feature may be soil type.
#
# Make sure to input all of the files you want to upload at once.
#
# > If you plan to re-run this workflow with the same data, you can expand the cell and and the paths to the `value` parameter, which sets a default value. For example, `value = '...data/feature1.tif,...data/feature2.tif'`.
#
# ### Mask
#
# A mask defines the output the model learns to predict. For example, if you want to predict the land cover of a region, the masks would be polygons representing the land cover types.
#
# As of right now, this workflow only supports the ability to predict a single mask. 
#
# You can only upload a single **vector** file.

# %%
# add the paths to your data files

add_features = widgets.Textarea(
    value="../chile_data/alos_nova_friburgo_5m.tif,../chile_data/2328825_2011-08-13_RE1_3A_Analytic.tif",
    # value="../california_data/dem.tif,../california_data/tri.tif,../california_data/slope.tif,../california_data/roughness.tif",
    placeholder="File paths (separated by commas)",
    description="Features:",
    disabled=False,
)

add_mask = widgets.Textarea(
    value="../chile_data/scars.geojson",
    # value='../california_data/landslide_deposits.gpkg',
    placeholder="File path",
    description="Mask:",
    disabled=False,
)

tabs = widgets.Tab()
tabs.children = [add_features, add_mask]
tabs.set_title(0, "Input Feature Paths")
tabs.set_title(1, "Input Mask Path")

tabs

# %%
# # copy the input data files to the working directory
mask_path = shutil.copy2(
    add_mask.value, os.path.join(working_dir, pathlib.Path(add_mask.value).name)
)
print(f"{time.ctime()}: Saved a working copy of the mask to {mask_path}.")

feature_paths = []
for f in add_features.value.split(","):
    path = os.path.join(working_dir, pathlib.Path(f).name)
    print(f"{time.ctime()}: Saved a working copy of the feature to {path}.")
    feature_paths.append(path)
    try:
        shutil.copy2(f, path)
    except:
        pass  # ignore if same file

# %% [markdown]
# ## Set the No Data Value

# %%
select_nodata = widgets.IntText(value=-1, description="no_data:", disabled=False)

select_nodata

# %% [markdown]
# ## Select a Coordinate Reference System (CRS)
#

# %% [markdown]
# Input data does not all need to have the same CRS, it will be reprojected to the CRS selected here. The CRS of your mask input will be set to the default value. 
#
# This workflow can use any CRS accepted by the function [`pyproj.CRS.from_user_input()`](https://geopandas.org/en/stable/docs/user_guide/projections.html):
#
# - CRS WKT string
# - An authority string (i.e. "EPSG:4326")
# - An EPSG integer code (i.e. 4326)
# - A pyproj.CRS
# - An object with a to_wkt method
# - PROJ string
# - Dictionary of PROJ parameters
# - PROJ keyword arguments for parameters
# - JSON string with PROJ parameters
#
#
# For reference, some common projections and their codes:
#
# - WGS84 Latitude/Longitude: "EPSG:4326"
# - UTM Zones (North): "EPSG:32633"
# - UTM Zones (South): "EPSG:32733"
#
# <!-- TODO add details about what is a CRS (geographic vs projected), which to select, details about why there are limited options, etc -->

# %%
# select a CRS
NO_DATA = select_nodata.value

# set a base CRS
default = "EPSG:4326"

# find the CRS from the mask file
try:  # to open the mask file as a polygon
    tmp = gpd.read_file(mask_path)
    default = tmp.crs
except:  # try to open the mask file as a raster
    tmp = rio.open(mask_path)
    default = tmp.crs

print(f"{time.ctime()}: Detected CRS from {mask_path} to be {default}")

select_crs = widgets.Text(
    value=str(default).upper(), description="CRS:", disabled=False
)

select_crs

# %% [markdown]
# ## Reproject Data

# %% [markdown]
# If *necessary*, each of the input features and the mask will be reprojected to the CRS selected above. 
#
# If the following cell fails to run, try rerunning the cell. If you are still having issues, you may be using an incompatible CRS for your data.

# %%
# convert all data to the same CRS
CRS = select_crs.value  # get the selected CRS from the widget

min_res = -1  # set to negative 1
max_res = np.inf  # set to infinity

# reproject all data to the same CRS
for data in [mask_path, *feature_paths]:
    try:
        reproject_raster(data, data, CRS)
        # calculate the minimum resolution (worst resolution of the data)
        min_res = max(min_res, rio.open(data).res[0])
        # calculate the maximum resolution (best resolution of the data)
        max_res = min(max_res, rio.open(data).res[0])
    except:
        gdf = gpd.read_file(data)
        if gdf.crs == CRS:
            print(f"{time.ctime()}: {data} is already in target CRS.")
        else:
            gdf.to_crs(CRS, inplace=True)
            os.remove(data)
            gdf.to_file(data)

# %% [markdown]
# ## Determine the bounds of the area of interest
#
# <!-- Bounds define a rectangular area of interest.
#
# You may input the bounds manually or it will be assumed that the bounds are equivalent to the extent of the mask data. -->
#
# <!-- ### Upload a file describing the bound -->
#

# %% [markdown]
# The bounds of the area of interest can be determined in three ways: 
#
# 1. DEFAULT: The bounds will be determined by a **concave hull polygon** of the mask data.
# 2. Automatic determination of the **total bounds** (rectangle that encompasses all mask polygons). For more information, please see the [documentation](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoSeries.total_bounds.html).
# 3. Upload a custom area definition in the form of a vector file. 

# %%
# select bounds determination method

options = ["Concave Hull (default)", "Convex Hull", "Total Bounds", "Custom Bounds"]
select_bounds_method = widgets.Dropdown(
    value=options[0], description="Method:", options=options, disabled=False
)

select_bounds_method

# %%
# calculate the bounds

mask_polygons = gpd.read_file(mask_path)
if select_bounds_method.value == options[0]:
    # concave hull of the mask region
    bounds_gpd = concave_hull(mask_polygons)

    # rectangular max bounds of the mask region (for cropping)
    total_bounds = mask_polygons.total_bounds
    total_bounds_poly = box(*total_bounds)
    total_bounds_gs = gpd.GeoSeries(total_bounds_poly, crs=CRS)

elif select_bounds_method.value == options[1]:
    # convex hull of the mask region
    convex_hull = mask_polygons[mask_polygons.is_valid].unary_union.convex_hull
    bounds_gpd = gpd.GeoDataFrame(geometry=[convex_hull], crs=CRS)

    # rectangular max bounds of the mask region (for cropping)
    total_bounds = mask_polygons.total_bounds
    total_bounds_poly = box(*total_bounds)
    total_bounds_gs = gpd.GeoSeries(total_bounds_poly, crs=CRS)

elif select_bounds_method.value == options[2]:
    total_bounds = mask_polygons.total_bounds
    total_bounds_poly = box(*total_bounds)
    total_bounds_gs = gpd.GeoSeries(total_bounds_poly, crs=CRS)
    bounds_gpd = total_bounds_gs

else:
    # creata a widget to select custom bounds
    upload_bounds = widgets.Text(
        value="Enter path...", description="CRS:", disabled=False
    )

    # display the widget
    display(upload_bounds)

# %% [markdown]
# ### Visualize the bounds
#

# %% [markdown]
# The bounds will be plotted on a basemap (if available) for reference. Specifically, Esri's National Geographic World Map will be used.
#
# - The bounds will be plotted as a blue polygon.
# - The masks will be plotted in red for reference. 
#
# A GEOJSON file describing the determined bounds will be saved to the working directory. This file can be used to visualize the bounds in a GIS software such as QGIS or ArcGIS.
#
# > If your area of interest is small, the background reference map may not load--this does not affect the workflow, the basemap is only provided at this step for reference.

# %%
# plot the bounds

# custom bounds
if select_bounds_method.value == options[2]:
    # read bounds from file
    bounds_gpd = gpd.read_file(upload_bounds.value)

    # reproject bounds to the CRS
    bounds_gpd = bounds_gpd.to_crs(CRS)

    # get the total bounds for cropping
    total_bounds = bounds_gpd.total_bounds
    total_bounds_poly = box(*total_bounds)


# save the bounds to file
bounds_gpd.to_file(os.path.join(working_dir, "bounds.geojson"))
print(f'{time.ctime()}: Saved bounds to {os.path.join(working_dir, "bounds.geojson")}')

# plot the bounds
bounds_ax = bounds_gpd.boundary.plot(figsize=(8, 8))

# add a title to the plot
bounds_ax.set_title("Bounds")

# add axis labels
bounds_ax.set_xlabel("Longitude")
bounds_ax.set_ylabel("Latitude")

# add the masks to the plot
mask_polygons.plot(ax=bounds_ax, color="red")

# add a basemap to the plot
cx.add_basemap(bounds_ax, source=cx.providers.Esri.NatGeoWorldMap, crs=CRS)

# %%
total_area = bounds_gpd.area[0]
unit = bounds_gpd.crs.axis_info[0].unit_name
print(
    f"{time.ctime()}: Total area of the bounds is {total_area:.2f} in square {unit}s."
)

obj_area = mask_polygons.area.sum()
print(f"{time.ctime()}: Total area of the mask is {obj_area:.2f} square {unit}s.")

ratio = obj_area / total_area
print(f"{time.ctime()}: Ratio of mask to bounds is {ratio:.3f}.")

CLASS_WEIGHTS = {0: 1 - ratio, 1: ratio}

# %% [markdown]
# # Data Preparation
#

# %% [markdown]
# ## Select tile size & data resolution
#
# <!-- ### What is a tile size?
#
# ### Why are tile sizes powers of 2?
#
# ### How to choose a tile size -->
#

# %% [markdown]
# The tile size is the size of the image patches in `tile_size * tile_size` pixels that will be extracted from the input data. 
#
# The resolution is automatically determined by the input data, but can be modified.
#
# <!-- It is not recommended increase the resolution of the data -->

# %%
# set tile size and data resolution
select_tile_size = widgets.Dropdown(
    options=[16, 32, 64, 128, 256, 512],
    value=64,
    description="Tile Size:",
    disabled=False,
)

print("For reference:")
print(
    f"{time.ctime()}: Detected minimum (lowest) resolution of {min_res} m from input data."
)
print(
    f"{time.ctime()}: Detected maximum (highest) resolution of {max_res} m from input data."
)

select_res = widgets.BoundedFloatText(
    # value=0.00010,  # defaults to 10 m
    min=max_res,
    step=0.00001,  # 1 m
    description="Resolution:",
    disabled=False,
)

# combine the widgets into an HBox layout
widget_layout = widgets.HBox([select_tile_size, select_res])

# display the layout
widget_layout

# %% [markdown]
# ## Select vector file features
#
# If you uploaded any features in vector file format, you will be prompted to select which features to use. Vector files may include multiple features, which will need to be encoded as separate bands in the composite/stacked raster used to train the model. If you don't select any features, the function will default to selecting the first feature it finds in the first vector file it searches. 
#
# It is recommended that you only select the most relevant features to reduce the time and space complexity of the model (how long and how much memory it takes to run). Additionally, too many inputs may cloud the model's ability to learn the relationship between the inputs and the mask.
#
# > Note that this is not relevant for raster feature files, which will be processed later. 

# %%
# select features to train on

# set tile & pixel size from widget selections
TILE_SIZE = select_tile_size.value
PIXEL_SIZE = select_res.value

all_col = []

for path in feature_paths:
    # print(path, feature)
    feature = Path(os.path.basename(path)).stem
    try:  # try to open aka check if its a vector
        gdf = gpd.read_file(path)
        all_col.extend([f"{feature}: {n}" for n in gdf.columns])
        all_col.remove(f"{feature}: geometry")
    except:
        print(
            f"{time.ctime()}: Feature input {path} is already a raster file, no features need to be extracted."
        )

# select columns to keep
to_keep = widgets.SelectMultiple(
    options=all_col,
    # value=[all_col[0]], # defaults to first feature
    description="Features: ",
    disabled=False,
)

if len(all_col) > 0:
    to_keep.value = [all_col[0]]  # defaults to first feature
    display(to_keep)
else:
    print(f"{time.ctime()}: You do not need to select any input features.")

# %% [markdown]
# ## Rasterize data
#

# %% [markdown]
# In order for the model to learn from the data, the input feature data will be encoded in raster bands. Therefore, any input features in vector file format will be rasterized using the data resolution selected above.
#
# > Note: Depending on the size of your data, this step may take several minutes to run. 

# %%
# rasterize masks and selected features if necessary
try:
    gdf = gpd.read_file(mask_path)
    old_path = mask_path
    mask_path = os.path.join(working_dir, "mask.tif")
    gdf["encoding_key"] = 1
    os.remove(old_path)
    gdf.to_file(old_path)
    to_raster(
        old_path, mask_path, feature_id="encoding_key", pixel_size=select_res.value
    )
except:
    print(f"{time.ctime()}: {mask_path} is already a raster file.")

# get the feature_ids for the features that need to be turned into bands
keeping = [re.findall(r"\s(.*)", s) for s in to_keep.value]
keeping = list(itertools.chain.from_iterable(keeping))
keeping = list(map(str.lower, keeping))

band_paths = []

# convert the features to rasters if necessary
for feature_path in feature_paths:
    try:
        gdf = gpd.read_file(feature_path)
        gdf.columns = list(map(str.lower, gdf.columns))

        for feature in keeping:
            if feature in gdf.columns:
                fn = feature + ".tif"
                to_raster(
                    feature_path,
                    os.path.join(working_dir, fn),
                    feature_id=feature,
                    pixel_size=PIXEL_SIZE,
                )
                band_paths.append(os.path.join(working_dir, fn))
                print(
                    f"{time.ctime()}: {feature} is in {feature_path}, band saved as {fn}."
                )
            else:
                print(f"{time.ctime()}: {feature} is not in {feature_path}")
    except:
        print(f"{time.ctime()}: {feature} is already a raster file.")
        band_paths.append(feature_path)

# %% [markdown]
# ## Stack the features into a multiband raster
#

# %%
# make sure all rasters are the same resolution & have x & y dims divisible by 2
for raster in tqdm([mask_path, *band_paths]):
    r = gdal.Open(raster)
    gdal.Warp(
        raster,
        r,
        xRes=PIXEL_SIZE,
        yRes=PIXEL_SIZE,
        resampleAlg="bilinear",
        multithread=True,
        copyMetadata=True,
        targetAlignedPixels=True,
        outputBounds=total_bounds,
        srcNodata=NO_DATA,
        dstNodata=NO_DATA,
    )

# crop to bounds and save to working directory
band_paths_list = es.crop_all(
    [mask_path, *band_paths], working_dir, total_bounds_gs, overwrite=True
)

# build a list describing the bands to be stacked in the composite image
band_paths = []
mask_path = ""
for path in band_paths_list:
    if "mask" in path:
        mask_path = path
    else:
        band_paths.append(path)

# create a save path for the stacked bands in the working dir
stack_path = os.path.join(working_dir, "stack.tif")

# %%
# read metadata of first band
with rio.open(band_paths[0]) as src:
    meta = src.meta

# count the number of bands in the input layers
band_count = 0
for layer in band_paths:
    with rio.open(layer, "r") as src:
        band_count += src.count

# update meta to reflect the number of layers
meta.update(count=band_count)

# read each layer/band and write it to stack using rasterio
BANDS = {}
id = 1
with rio.open(stack_path, "w", **meta) as dst:
    for layer in band_paths:
        with rio.open(layer, "r") as src:
            for band in range(1, src.count + 1):
                l = Path(layer).stem
                print(
                    f"{time.ctime()}: Writing band {band} from {l} to composite raster."
                )
                dst.write_band(id, src.read(band))
                BANDS[f"{os.path.basename(l)}_BAND-{band}"] = id - 1
                id += 1
        src.close()

print(f"{time.ctime()}: Composite raster saved as {stack_path}.")

# %% [markdown]
# ## Data Tiling
#

# %% [markdown]
# ### Create tiles

# %%
# create a spatial data generator for the mask and data stack
mask_sgd = SpatialDataGenerator(source=mask_path, interleave="pixel")
data_sgd = SpatialDataGenerator(source=stack_path, interleave="pixel")

# create regularly gridded tiles that cover the bounds of the mask w no overlap
tile_bounds_gdf = mask_sgd.regular_grid(TILE_SIZE, TILE_SIZE, overlap=0, units="pixels")

# create a geodataframe of the bounds of the mask
tiles_gdf = tile_bounds_gdf[tile_bounds_gdf.intersects(bounds_gpd.unary_union)].copy()

# print the number of tiles created
print(f"{time.ctime()}: Created {len(tiles_gdf)} tiles size {TILE_SIZE}x{TILE_SIZE}.")

# plot the tiles
tiles_ax = tiles_gdf.boundary.plot(figsize=(8, 8), edgecolor="black", linewidth=0.5)

# add a title to the plot
tiles_ax.set_title("Tiled Area of Interest")

# add axis labels
tiles_ax.set_xlabel("Longitude")
tiles_ax.set_ylabel("Latitude")

# plot the mask polygons underneath the tiles
mask_polygons.plot(ax=tiles_ax, color=None, edgecolor="red")

# plot the bounds of the area of interest
bounds_gpd.boundary.plot(ax=tiles_ax, color=None, linewidth=2)

# add a basemap to the plot
cx.add_basemap(tiles_ax, source=cx.providers.Esri.NatGeoWorldMap, crs=CRS)


# %% [markdown]
# ### Extract images and masks from tiles

# %%
# reshape the data to be in the correct shape for the model
def img_reshape(arr):
    return arr.reshape(TILE_SIZE, TILE_SIZE, -1)


# ensure that the data is in the correct shape for the model
data_sgd.add_preprocess_callback("reshape", img_reshape)
mask_sgd.add_preprocess_callback("reshape", img_reshape)

# create generators for the input data
X_gen = data_sgd.flow_from_dataframe(tiles_gdf, TILE_SIZE, TILE_SIZE, batch_size=1)
Y_gen = mask_sgd.flow_from_dataframe(tiles_gdf, TILE_SIZE, TILE_SIZE, batch_size=1)

# n = len(tiles_gdf) # number of tiles

# define a function to unpack a generator
def unpack_gen(gen):
    stack = []  # create an empty list to store the data
    while True:  # loop until the generator is exhausted
        try:
            stack.append(next(gen))  # append the next batch (1) to the list
        except StopIteration:
            break
    return np.stack(stack, axis=1)  # stack the list of batches into a single array


X = unpack_gen(X_gen)  # unpack the data generator
Y = unpack_gen(Y_gen)  # unpack the mask generator

# reshape the data to be in the correct format for the model
X = X[0, ...].astype(np.float32)
Y = Y[0, ...].astype(np.float32)

Y = np.where(Y > 0, 1, 0)  # binarize the mask

# print(X.shape, Y.shape)
# np.unique(Y)

# %% [markdown]
# ## Data Visualization
#
# Check to make sure things look good visually
#

# %%
# create three dropdowns to select the band to display
options = list(BANDS.keys())
options.insert(0, "None")
band1 = widgets.Dropdown(options=options, description="Band 1")
band2 = widgets.Dropdown(options=options, description="Band 2")
band3 = widgets.Dropdown(options=options, description="Band 3")

# display the dropdowns in boxes
b = widgets.VBox([band1, band2, band3])
display(b)

# %%
bands = []
for c in b.children:
    if c.value != "None":
        bands.append(BANDS[c.value])

r = 3
if not len(bands) == 0:
    n = np.random.randint(0, len(X), r)

    # plot r random tiles in separate images
    for i in range(r):
        spectral.imshow(
            X[n[i], ...],
            bands=bands,
            stretch=True,
            title=f"Tile {n[i]} with Bands {bands}",
            figsize=(3, 3),
        )

# %%
# display 10 random images with their masks
fig = plt.figure(figsize=(16, 16))
for i in range(5):
    n = np.random.randint(0, len(X))

    # mask
    ax = fig.add_subplot(1, 5, i + 1)
    ax.imshow(Y[n, :, :, 0], cmap="Greys_r")
    ax.set_title(f"Mask {n}")

    # image
    ax = fig.add_subplot(2, 5, i + 1)
    ax.imshow(X[n, :, :, 3], cmap="viridis")
    ax.set_title(f"Image {n}")

# %% [markdown]
# ## Normalize Data

# %%
# convert the data to the correct data type
X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.uint8)


def normalize_images(images):
    normalized_images = np.zeros_like(images, dtype=np.float32)
    num_images, _, _, channels = images.shape

    # Reshape the images to a 2D array for vectorized normalization
    reshaped_images = images.reshape(num_images, -1)

    # Perform per-channel normalization to [0, 1] range
    for c in range(channels):
        channel = images[:, :, :, c]
        normalized_channel = cv2.normalize(
            channel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        normalized_images[:, :, :, c] = normalized_channel

    return normalized_images


# normalize the images
X_norm = normalize_images(X)

print(
    f"{time.ctime()}: Completed per-channel, per-image normalization of {len(X_norm)} images."
)

# %% [markdown]
# ## Split Data & Set Parameters
#
# - add info about deciding split ratios and the other params to be selected

# %%
train_size = widgets.FloatSlider(
    value=0.65,
    min=0.00,
    max=1.0,
    step=0.01,
    description="Training %:",
    disabled=False,
    continuous_update=True,
    # orientation="vertical",
    readout=True,
    readout_format=".2f",
)

validation_size = widgets.FloatSlider(
    value=0.25,
    min=0.00,
    max=1.0,
    step=0.01,
    description="Validation %:",
    disabled=False,
    continuous_update=True,
    # orientation="vertical",
    readout=True,
    readout_format=".2f",
)

test_size = widgets.FloatSlider(
    value=0.10,
    min=0.00,
    max=1.0,
    step=0.01,
    description="Test %:",
    disabled=False,
    continuous_update=True,
    # orientation="vertical",
    readout=True,
    readout_format=".2f",
)

select_epochs = widgets.IntSlider(
    value=25,
    min=1,
    max=100,
    step=1,
    description="# Epochs:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format="d",
)

# should be a factor of 2 to take advantage of the GPU resources
select_batch_size = widgets.Dropdown(
    options=[8, 16, 32, 64, 128],
    value=32,
    description="Batch Size:",
    disabled=False,
)

widgets.VBox([train_size, validation_size, test_size, select_epochs, select_batch_size])

# %%
EPOCHS = select_epochs.value  # default = 50
BATCH_SIZE = select_batch_size.value  # default = 32

# validate the splits
# assert int(train_size.value + validation_size.value + test_size.value) == 1

TRAIN_SIZE = train_size.value
VALID_SIZE = validation_size.value
TEST_SIZE = test_size.value

# train test split
print(f"{time.ctime()}: Splitting data into train, validation, and test sets...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X_norm, Y, test_size=TEST_SIZE, shuffle=True, random_state=SEED
)

# extricate the validation set from the training set
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train, Y_train, test_size=VALID_SIZE, shuffle=True, random_state=SEED
)

# %%
print(f"{time.ctime()}: Dataset sizes:")
print(f"Training: {len(X_train)}")
print(f"Validation: {len(X_val)}")
print(f"Test: {len(X_test)}")

# %% [markdown]
# ## Data Augmentation
#
# - add info about deciding augmentation params

# %%
fliplr = widgets.FloatSlider(
    value=0.5,
    min=0.00,
    max=1.0,
    step=0.01,
    description="Flip L/R %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

flipud = widgets.FloatSlider(
    value=0.5,
    min=0.00,
    max=1.0,
    step=0.01,
    description="Flip U/D %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

transX = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Translate X %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

transY = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Translate Y %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

scaleX = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Scale X %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

scaleY = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Scale Y %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

add_noise = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Noise %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

hist_eq = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.01,
    description="Hist. Eq. %:",
    disabled=False,
    continuous_update=True,
    orientation="horizontal",
    readout=True,
    readout_format=".2f",
)

box = widgets.VBox([fliplr, flipud, transX, transY, scaleX, scaleY, hist_eq])

box


# %%
def filter_images_with_mask(X, Y):
    filtered_X = []
    filtered_Y = []

    for i in range(len(X)):
        if np.any(Y[i] != 0):  # Check if any pixel in the mask is non-zero
            filtered_X.append(X[i])
            filtered_Y.append(Y[i])

    return filtered_X, filtered_Y


X_pos, Y_pos = filter_images_with_mask(X_train, Y_train)

print(f"{time.ctime()}: Found {len(X_pos)} images with masks.")

# %%
# augment all images or just the ones with masks (objs of interest)
augment_all = widgets.RadioButtons(
    options=["Augment only images with masks", "Augment all images"],
    description="Augment all images or just the ones with masks?",
    disabled=False,
)

# choose whether to concatenate augmented or replace the original images
concat_replace = widgets.RadioButtons(
    options=["Concatenate", "Replace"],
    description="Replace or concatenate to original dataset?",
    disabled=False,
)

widgets.VBox([augment_all, concat_replace])

# %%
ia.seed(SEED)

seq = iaa.Sequential(
    [
        iaa.Fliplr(fliplr.value),  # LR flip
        iaa.Flipud(flipud.value),  # UD flip
        iaa.Sometimes(
            transX.value, iaa.TranslateX(percent=(-0.1, 0.1))
        ),  # x translation
        iaa.Sometimes(
            transY.value, iaa.TranslateY(percent=(-0.1, 0.1))
        ),  # y translation
        iaa.Sometimes(scaleX.value, iaa.ScaleX(scale=(0.8, 1.2))),  # x scale
        iaa.Sometimes(scaleY.value, iaa.ScaleY(scale=(0.8, 1.2))),  # y scale
        # iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255), per_channel=True),  # add gaussian noise
        # iaa.Sometimes(
        #     hist_eq.value, iaa.AllChannelsHistogramEqualization()
        # ),  # histogram equalization
    ]
)

if augment_all.value == "Augment all images":
    print(f"{time.ctime()}: Augmenting all images...")
    X_aug, Y_aug = seq(images=X, segmentation_maps=Y)
else:
    print(f"{time.ctime()}: Augmenting images with masks...")
    X_aug, Y_aug = seq(images=X_pos, segmentation_maps=Y_pos)

if concat_replace.value == "Concatenate":
    print(f"{time.ctime()}: Concatenating augmented images to original dataset...")
    X = np.concatenate((X, X_aug), axis=0)
    Y = np.concatenate((Y, Y_aug), axis=0)
else:
    print(f"{time.ctime()}: Replacing original dataset with augmented images...")
    X = X_aug
    Y = Y_aug

print(
    f"{time.ctime()}: Augmentation completed. The dimension of the augmented training dataset is {X.shape}."
)

# %% [markdown]
# ## Convert to *tf.data.Dataset* format

# %%
# transform the numpy arrays into tensorflow datasets
print(f"{time.ctime()}: Transforming numpy arrays into tensorflow datasets...")
train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
valid_ds = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test))

print(f"{time.ctime()}: Batching and parallelizing the datasets...")
# batch the data, drop the remainder, & parallelize according to available resources
train_ds = train_ds.batch(
    BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
)
valid_ds = valid_ds.batch(
    BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(
    BATCH_SIZE, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
)

# use all of data points, 1 batch size at a time
STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
print(f"{time.ctime()}: Calculated steps per epoch to be: {STEPS_PER_EPOCH}")

# calculate validation steps
VALIDATION_STEPS = len(X_val) // BATCH_SIZE
print(f"{time.ctime()}: Calculated validation steps to be: {VALIDATION_STEPS}")

print(f"{time.ctime()}: Batching and parallelization completed.")

# %% [markdown]
# # Train the Model
#

# %% [markdown]
# ## Model Architecture

# %%
# # build a unet
# def bottleneck(inputs, n):
#     conv = Conv2D(n, (3, 3), activation="relu", padding="same")(inputs)
#     conv = Conv2D(n, (3, 3), activation="relu", padding="same")(conv)

#     return conv


# def downsample(inputs, n):
#     conv = Conv2D(n, (3, 3), activation="relu", padding="same")(inputs)
#     conv = Conv2D(n, (3, 3), activation="relu", padding="same")(conv)
#     pool = MaxPooling2D((2, 2))(conv)
#     # pool = Dropout(0.25)(pool)

#     return conv, pool, n * 2


# def upsample(inputs, residual, n):
#     n = n // 2
#     deconv = Conv2DTranspose(n, (3, 3), strides=(2, 2), padding="same")(inputs)
#     uconv = concatenate([deconv, residual])
#     # uconv = Dropout(0.5)(uconv)
#     uconv = Conv2D(n, (3, 3), activation="relu", padding="same")(uconv)
#     uconv = Conv2D(n, (3, 3), activation="relu", padding="same")(uconv)

#     return uconv, n


# def build_unet(input_shape, n=32):
#     inputs = Input(shape=input_shape)

#     # downsample
#     conv1, pool1, n = downsample(inputs, n)
#     conv2, pooreg, n = downsample(pool1, n)
#     conv3, pool3, n = downsample(pool2, n)
#     conv4, pool4, n = downsample(pool3, n)

#     # bottleneck
#     conv5 = bottleneck(pool4, n)

#     # upsample
#     uconv4, n = upsample(conv5, conv4, n)
#     uconv3, n = upsample(uconv4, conv3, n)
#     uconv2, n = upsample(uconv3, conv2, n)
#     uconv1, n = upsample(uconv2, conv1, n)

#     outputs = Conv2D(1, (1, 1), activation="sigmoid")(uconv1)

#     return Model(inputs=[inputs], outputs=[outputs])

# %%
# unet w batch norm & dropout
reg = tf.keras.regularizers.l1_l2()

def double_conv_block(x, n_filters):
    # conv2D then ReLU activation
    x = Conv2D(
        n_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        # kernel_regularizer=reg,
    )(x)
    # conv2D then ReLU activation
    x = Conv2D(
        n_filters,
        3,
        padding="same",
        activation="relu",
        kernel_initializer="he_normal",
        # kernel_regularizer=reg,
    )(x)

    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)  # feature map
    f = BatchNormalization()(f)  # batch normalization
    p = MaxPool2D(2)(f)  # pooled feature map
    p = Dropout(0.3)(p)  # dropout

    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = UpSampling2D()(x)
    x = Conv2D(
        n_filters,
        2,
        padding="same",
        kernel_initializer="he_normal",
        # kernel_regularizer=reg,
    )(x)
    # x = Conv2DTranspose(filters=n_filters, kernel_size=3, strides=(2, 2), padding="same")(x)
    # concatenate
    x = concatenate([x, conv_features])
    # dropout
    x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    # batch normalization
    x = BatchNormalization()(x)

    return x


def build_unet(input_shape, output_channels, depth=5, base_filters=64):
    # inputs
    inputs = Input(shape=input_shape)

    # encoder: contracting path - downsample
    skips = []
    filters = []
    p = inputs
    for i in range(depth):
        nf = base_filters * 2**i
        f, p = downsample_block(p, nf)
        skips.append(f)
        filters.append(nf)

    # bottleneck
    bottleneck = double_conv_block(p, base_filters * 2**depth)

    # decoder: expanding path - upsample
    u = bottleneck
    for i in reversed(range(depth)):
        u = upsample_block(u, skips[i], filters[i])

    # check if output channels is 1 (binary) or > 1 (multiclass)
    if output_channels == 1:
        a_func = "sigmoid"
        print(
            f"{time.ctime()}: Using sigmoid activation function for binary semantic segmentation."
        )
    else:
        a_func = "softmax"
        print(
            f"{time.ctime()}: Using softmax activation function for multiclass semantic segmentation."
        )

    # outputs
    outputs = Conv2D(output_channels, 1, padding="same", activation=a_func)(u)

    # U-Net model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    return unet_model


# %%
import tensorflow as tf
from tensorflow.keras import layers


class Net(object):
    def __init__(
        self,
        input_shape,
        output_channels,
        filters=32,
        depth=3,
        dilation_rates=[1],
        reduction_ratio=16,
        name="model",
    ):
        self.input_shape = input_shape
        self.output_channels = output_channels
        self.filters = filters
        self.depth = depth
        self.dilation_rates = dilation_rates
        self.reduction_ratio = reduction_ratio
        self.name = name
        self.skips = []
        self.inputs = tf.keras.Input(shape=input_shape)

    def channel_attention(self, inputs):
        # channel attention
        channels = inputs.shape[-1]
        avg_pool = layers.GlobalAveragePooling2D()(inputs)
        fc1 = layers.Dense(channels // self.reduction_ratio)(avg_pool)
        fc1 = layers.ReLU()(fc1)
        fc2 = layers.Dense(channels)(fc1)
        fc2 = layers.Activation("sigmoid")(fc2)
        reshaped = layers.Reshape((1, 1, channels))(fc2)
        return layers.Multiply()([inputs, reshaped])

    def spatial_attention(self, inputs, kernel_size=5):
        # spatial attention
        spatial_attn = layers.Conv2D(1, kernel_size, padding="same")(inputs)
        spatial_attn = layers.Activation("sigmoid")(spatial_attn)
        return layers.Multiply()([inputs, spatial_attn])

    def CBAM_block(self, inputs):
        # conv
        x = layers.Conv2D(self.filters, 3, padding="same")(inputs)
        # bn + relu
        x = layers.BatchNormalization()(x)
        # channel attention
        channel_attention_output = self.channel_attention(inputs, self.reduction_ratio)
        # spatial attention
        spatial_attention_output = self.spatial_attention(inputs)
        # merge
        output = layers.Add()([channel_attention_output, spatial_attention_output])
        return output

    def pyramid_pooling_block(self, inputs, bin_sizes=[1, 2, 3, 6]):
        h, w, c = inputs.shape[1:]
        pooled_outputs = []  # store pooled output tensors for each bin size
        # remove invalid bin sizes
        bin_sizes = [size for size in bin_sizes if min(h, w) % size == 0]
        # print("bin sizes", bin_sizes)
        for size in bin_sizes:  # iterate over each bin size
            x = layers.AveragePooling2D(pool_size=(h // size, w // size))(inputs)
            x = layers.Conv2D(self.filters, 1, padding="same")(x)
            x = layers.ReLU()(x)
            # resize so all pooled output tensors have the same shape
            x = layers.Reshape((h, w, -1))(x)
            pooled_outputs.append(x)
        # concatenate pooled output tensors
        x = layers.Concatenate(axis=-1)(pooled_outputs)
        # conv block
        x = layers.Conv2D(self.filters, 3, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        # conv block
        x = layers.Conv2D(self.filters, 1, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)
        return x

    # residual block with atrous convolutions
    def residual_block(self, inputs):
        x = inputs
        stack = [x]
        for rate in self.dilation_rates:
            # conv block 1
            x = layers.Conv2D(self.filters, 3, padding="same", dilation_rate=rate)(x)
            x = layers.ReLU()(x)
            x = layers.BatchNormalization()(x)
            # conv block 2
            x = layers.Conv2D(self.filters, 1, padding="same", dilation_rate=rate)(x)
            x = layers.ReLU()(x)
            x = layers.BatchNormalization()(x)
            # apply channel-wise attention
            se = self.channel_attention(x)
            # add to stack to combine parallel atrous convolutions
            stack.append(se)
        # print([s.shape for s in stack])
        x = layers.Add()(stack)
        return x

    def encoding_block(self, inputs):
        # residual block
        x = self.residual_block(inputs)
        # enhance skip with spatial attention
        skip = self.spatial_attention(x)
        # update filters
        self.filters *= 2
        # downsampling with conv2d
        x = layers.Conv2D(self.filters, 1, 2, padding="same")(skip)
        return x, skip

    def decoding_block(self, inputs):
        # calculate new number of filters
        self.filters //= 2
        # upsample the feature maps
        x = layers.UpSampling2D()(inputs)
        # concatenate skip connection from the corresponding encoding block
        x = layers.Concatenate(axis=-1)([x, self.skips.pop()])
        # residual block
        stack = []
        for rate in self.dilation_rates:
            # conv block 1
            x = layers.Conv2D(self.filters, 3, padding="same", dilation_rate=rate)(x)
            x = layers.ReLU()(x)
            x = layers.BatchNormalization()(x)
            # conv block 2
            x = layers.Conv2D(self.filters, 1, padding="same", dilation_rate=rate)(x)
            x = layers.ReLU()(x)
            x = layers.BatchNormalization()(x)
            # apply channel-wise attention
            se = self.channel_attention(x)
            # add to stack to combine parallel atrous convolutions
            stack.append(se)
        # print([s.shape for s in stack])
        x = layers.Add()(stack)
        return x

    def bridge_block(self, inputs):
        # conv block
        x = layers.Conv2D(self.filters, 3, padding="same")(inputs)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)

        # psp block
        # x = self.PSPPooling_block(inputs)

        # conv block
        x = layers.Conv2D(self.filters, 3, padding="same")(x)
        x = layers.ReLU()(x)
        x = layers.BatchNormalization()(x)

        # cbam block
        # x = self.CBAM_block(x)

        # residual block
        # x = self.residual_block(x)
        return x

    def bridge_block(self, inputs):
        x = self.pyramid_pooling_block(inputs)
        # x = self.pyramid_pooling_block(x)
        return x

    def build(self):
        # input block - inc to initial desired filter size
        x = self.inputs
        x = layers.Conv2D(self.filters, kernel_size=(1, 1), padding="same")(x)
        x = self.spatial_attention(x)
        self.skips.append(x)

        x = self.residual_block(x)

        # encoding path
        for _ in range(self.depth):
            # print(self.filters, x.shape)
            x, skip = self.encoding_block(x)
            self.skips.append(skip)

        # bridge
        # print("bridge", self.filters)
        x = self.bridge_block(x)
        # x = self.CBAM_block(x)

        # decoding path
        for _ in range(self.depth):
            x = self.decoding_block(x)
            # print(self.filters, x.shape)

        # check if multiclass or binary classification
        if self.output_channels > 1:
            a_fn = "softmax"
        else:
            a_fn = "sigmoid"

        # psp block

        # get final skip connection from stack
        x = layers.Concatenate(axis=-1)([x, self.skips.pop()])
        # output segmentation map
        outputs = layers.Conv2D(
            self.output_channels, 1, padding="same", activation=a_fn
        )(x)
        print(self.input_shape, outputs.shape)
        return tf.keras.Model(inputs=self.inputs, outputs=outputs, name=self.name)

# %%
# todo add widgets for hyperparameter tuning (sliders galore)

# hyperparams to think about:

# - train, valid, test size, tile size, pixel size
# - epochs, batch size, optimizer, loss functions, accuracy metrics
# - call backs, early stopping, patience
# select loss function


# select optimizer


# select metrics

# %%
def dice_coefficient(y_true, y_pred, smooth=1e-5):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice_coeff = (2.0 * intersection + smooth) / (union + smooth)

    return dice_coeff


def weighted_dice_loss(class_proportions):
    def loss(y_true, y_pred):
        dice = dice_coefficient(y_true, y_pred)

        # compute weights based on class proportions
        weights = tf.constant(class_proportions, dtype=tf.float32)

        # index 0: background; 1: foreground
        weighted_dice = tf.multiply(y_true * weights[1], dice) + tf.multiply(
            (1.0 - y_true) * weights[0], dice
        )
        return 1.0 - tf.reduce_mean(weighted_dice)

    return loss


# %%
# define model parameters
# BATCH_SIZE = 32
INPUT_SHAPE = (TILE_SIZE, TILE_SIZE, band_count)  # X.shape[-1])

# compile model

# IF BINARY SEG, OUTPUT_CHANNELS = 1
# model = build_model(input_shape=INPUT_SHAPE, output_channels=1, base_filters=64, depth=4)
model = Net(INPUT_SHAPE, output_channels=1, filters=32, depth=6).build()
print(f"{time.ctime()}: Model compiled successfully.")

# define metrics
metrics = [
    "accuracy",
    # tf.keras.metrics.Accuracy(),
    # tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.BinaryIoU(),
    dice_coefficient,
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    # tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5),
    # tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2, threshold=0.5),
    tf.keras.losses.BinaryCrossentropy(),
]

# tf.keras.metrics.BinaryIoU(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives()]

# loss = tfa.losses.SigmoidFocalCrossEntropy()

class_weights = list(CLASS_WEIGHTS.values())

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryFocalCrossentropy(),
    # loss=weighted_dice_loss(class_weights),
    # loss_weights=class_weights,
    metrics=metrics,
)

# free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()
# tf.config.experimental.set_memory_growth(device='PhysicalDevice', enable=True)

# %%
tf.keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(working_dir, "model_architecture.png"))

# %% [markdown]
# ## Training

# %%
# todo add widget for setting output directory for model
output_dir = os.path.join(working_dir, "models")
file_name = f"model_{datetime.now()}"
file_name = file_name.replace(" ", "_").replace(":", "-").replace(".", "-")
model_path = os.path.join(output_dir, file_name + ".hdf5")

print(f"{time.ctime()}: Model will be saved to {model_path}")

# set up callbacks
checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_loss",
    mode="auto",
    save_best_only=True,
    restore_best_weights=True,
    save_freq="epoch",
    verbose=1,
)

# keep track of the model training progression
h = tf.keras.callbacks.History()

callbacks = [
    checkpoint,
    tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss", mode="auto"),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(working_dir, "logs")),
    h,
]

# %%
# print model training details
print(f"{time.ctime()}: Model training details:")
print(f"Model name: {model.name}")
print(f"Input shape: {INPUT_SHAPE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch: {STEPS_PER_EPOCH}")
print(f"Validation steps: {VALIDATION_STEPS}")
# print(f"Loss function: {model.loss}")
# print(f"Optimizer: {model.optimizer}")
# print(f"Metrics: {model.metrics}")

# %%
history = model.fit(
    train_ds,  # batch size is determined by the dataset
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_ds,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=1,
)

# %% [markdown]
# # Results
#

# %% [markdown]
# ## Calculate Performance Metrics
#

# %%
# model_path = "./working/models/model_2023-07-10_20-33-44-176831.hdf5"
best_model = tf.keras.models.load_model(
    model_path,
    custom_objects={
        "loss": weighted_dice_loss(class_weights),
        "dice_coefficient": dice_coefficient,
        "BinaryCrossentropy": tf.keras.losses.BinaryCrossentropy(),
    },
)  #                                        'focal_loss': semseglosses.focal_loss })

# use the model to make predictions on the reserved test data
score = best_model.evaluate(X_test, Y_test, verbose=1)
# print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# %% [markdown]
# ## Visualize Results

# %%
# Display n random images with their masks and predicted masks
random_indices = np.random.choice(range(len(X_test)), size=3, replace=False)

# Iterate over the randomly selected images
for idx in random_indices:
    # Get the original image (first three channels)
    original_image = X_test[idx][:, :, 4]

    # Get the true mask and predicted mask
    true_mask = Y_test[idx]
    predicted_mask = best_model.predict(X_test[idx].reshape(-1, 32, 32, 6))
    predicted_mask = np.squeeze(predicted_mask)
    predicted_mask = (predicted_mask > 0.2).astype(np.uint8)

    # Plotting the images
    fig, axs = plt.subplots(1, 3, figsize=(5, 5))
    axs[0].imshow(original_image)
    axs[0].set_title("Original Image")

    axs[1].imshow(true_mask, cmap="gray")
    axs[1].set_title("True Mask")

    axs[2].imshow(predicted_mask, cmap="gray")
    axs[2].set_title("Predicted Mask")

    plt.show()

# %%
END_TIME = time.time()
print(f"{time.ctime()}:  Elapsed time: {END_TIME - START_TIME}")
