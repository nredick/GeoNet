#!/usr/bin/env python
# coding: utf-8
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.10.7 (''.venv-geospatial_unet'': venv)'
#     language: python
#     name: python3
# ---

# %%

# ## Imports
# 

# %%


import os
import shutil
import time
from glob import glob
from os import path as osp

import contextily as cx
import earthpy.plot as ep
import earthpy.spatial as es
import geojson
import geopandas as gpd
import getuseragent as gua
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import pyrsgis
import rasterio
import requests
import richdem as rd
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from osgeo import gdal
from pandas_profiling import ProfileReport
from rasterio import features
from rasterio.plot import show
from shapely.geometry import *
import multiprocessing

from shapely.geometry import box
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Environment settings
# 

# %%


seed = 26
np.random.seed(seed)  # set a seed for reproducible results

CRS = "epsg:3857"
headers = {"User-Agent": gua.UserAgent('desktop').Random()}


# # Acquire data
# 

# %%


data_dir = '../data/working'

# create a working directory to hold data files
try:
    os.makedirs(data_dir, exist_ok=False)
except OSError:
    shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=False)


# ## Landslide data
# 
# Courtesy of the [California Geological Survey](https://www.conservation.ca.gov/cgs/).
# 

# %%


# copy landslide data to working directory
landslide_src_path = shutil.copy(
    '../data/CGS_Landslide_Inventory/landslide_sources.gpkg', osp.join(data_dir, 'landslide_sources.gpkg'))

landslide_dep_path = shutil.copy(
    '../data/CGS_Landslide_Inventory/landslide_deposits.gpkg', osp.join(data_dir, 'landslide_deposits.gpkg'))


# %%


# read the gpkg data using geopandas
scarps = gpd.read_file(landslide_src_path).to_crs(CRS)
deposits = gpd.read_file(landslide_dep_path).to_crs(CRS)


# %%





# ### Visualize landslide scarps and deposits
# 
# [Basemap documentation](https://contextily.readthedocs.io/en/latest/providers_deepdive.html)
# 

# %%


# plot the landslide sources
src_ax = scarps.plot(figsize=(8, 8), alpha=0.5, color='green', edgecolor='k')
src_ax.set_title('Landslide Scarps')
src_ax.set_xlabel('Longitude')
src_ax.set_ylabel('Latitude')
src_ax.grid(True)
cx.add_basemap(src_ax, source=cx.providers.Esri.WorldShadedRelief)


# %%


# plot the landslide deposits
dep_ax = deposits.plot(figsize=(8, 8), alpha=0.5, color='red', edgecolor='k')
dep_ax.set_title('Landslide Deposits')
dep_ax.set_xlabel('Longitude')
dep_ax.set_ylabel('Latitude')
dep_ax.grid(True)
cx.add_basemap(dep_ax, source=cx.providers.Esri.WorldShadedRelief)


# %%





# %%





# %%


# get the extent of the landslide data in different formats
lst_bounds = [max(scarps.total_bounds[i], deposits.total_bounds[i])
              for i in range(len(scarps.total_bounds))]
pg_bounds = box(*lst_bounds)
str_bounds = ",".join(str(x) for x in lst_bounds)

print(str_bounds)


# %%


bounds_poly = gpd.GeoSeries(pg_bounds)
bounds_poly = bounds_poly.set_crs(CRS)
bounds_ax = bounds_poly.plot(
    figsize=(8, 8), alpha=0.5, color='gray', edgecolor='red', linewidth=3)
bounds_ax.set_title('Research Area Extent')
cx.add_basemap(bounds_ax, source=cx.providers.Esri.WorldShadedRelief)


# %%


bounds_poly.to_file(osp.join(data_dir, 'research_extent.geojson'), driver="GeoJSON")


# ## Geological data
# 
# [Source](https://www.sciencebase.gov/arcgis/rest/services/Catalog/5888bf4fe4b05ccb964bab9d/MapServer/3): State Geologic Map Compilation ([SGMC](https://www.sciencebase.gov/catalog/item/5888bf4fe4b05ccb964bab9d)) geodatabase of the conterminous United States
# 

# %%


# retrieve the geology data from the CGS Mapserver API
response = requests.get('http://gis.conservation.ca.gov/server/rest/services/CGS/Geologic_Map_of_California/FeatureServer/12/query', params={
    "geometry": str_bounds,
    "geometryType": "esriGeometryEnvelope",
    "spatialRel": "esriSpatialRelIntersects",
    "where": "1=1",
    "units": "esriSRUnit_Meter",
    "outFields": "GENERAL_LITHOLOGY,AGE,DESCRIPTION",
    "returnGeometry": "true",
    "returnTrueCurves": "false",
    "returnIdsOnly": "false",
    "returnCountOnly": "false",
    "returnDistinctValues": "false",
    "returnExtentOnly": "false",
    "sqlFormat": "none",
    "featureEncoding": "esriDefault",
    "f": "geojson",
}, headers=headers)

print(response.status_code)


# %%


if response.status_code == 200:
    print(f'{time.ctime()} | Successfully retrieved geological data')
    # write the landslide data to a geojson file
    geology_path = osp.join(data_dir, 'geology.geojson')
    with open(geology_path, 'w') as f:
        geojson.dump(response.json(), f, indent=4)
else:
    print(f'{time.ctime()} | ERROR {response.status_code}: Failed to retrieve geological data')


# %%


# load the geology data into a geopandas dataframe and set the correct CRS
geology = gpd.read_file(geology_path).to_crs(CRS)
geology = geology.clip(mask=bounds_poly)
display(geology)

geo_ax = geology.plot(
    figsize=(15, 10), column='GENERAL_LITHOLOGY', legend=True, alpha=0.5, edgecolor='k')
geo_ax.set_title('Generalized Lithology of the Study Area')
cx.add_basemap(geo_ax, source=cx.providers.Esri.WorldShadedRelief)
leg = geo_ax.get_legend()


# %%


# create a statistical profile using the pandas profiling lib
profile = ProfileReport(pd.DataFrame(geology.drop(
    columns='geometry')), title="Geology Data Profile", explorative=False)

display(profile)


# The statistical report shows a nearly 1-1 correlation between the data features: lithology (_GENERAL_LITHOLOGY_), age (_AGE_), & the description (_DESCRIPTION_). While the description is helpful to a human reader, it is not informative for the model. Since our goal is to determine how rock type affects landslide risk, we will use the lithology feature rather than the age feature in the model.
# 

# %%


# remove the excess features
lithology = geology.drop(columns=['AGE', 'DESCRIPTION'], inplace=False)
lithology.dropna(inplace=True)


# %%


display(lithology)


# ## Elevation derivatives
# 
# DEM data from the USGS 3DEP database.
# 
# Slope, aspect curvature rasters calculated using [`richdem`](https://richdem.com).
# 

# %%


xmin, ymin, xmax, ymax = scarps.total_bounds
# corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]

delta_x = round(abs(xmax - xmin)/10, -2)
delta_y = round(abs(ymax - ymin)/10, -2)

# every pixel is 10m
print(delta_x, delta_y)


# %%


# Function to find the number closest to n and divisible by m
def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)

    # 1st possible closest number
    n1 = m * q

    # 2nd possible closest number
    if((n * m) > 0):
        n2 = (m * (q + 1))
    else:
        n2 = (m * (q - 1))

    # if true, then n1 is the required closest number
    if (abs(n - n1) < abs(n - n2)):
        return n1

    # else n2 is the required closest number
    return n2


# %%





# ### Set tile size

# %%


TILE_SIZE = 64  # set the tile size to TILE_SIZExTILE_SIZE pixels

# approx image size to be ~10m per pixel & divisible into tile_size X tile_size tiles
size_x = closestNumber(delta_x, TILE_SIZE)
size_y = closestNumber(delta_y, TILE_SIZE)


# ### Download DEM
# 

# %%


# download dem from 3DEPELEVATION imageserver
image_data = requests.get(url="https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage",
                          params={"bbox": str_bounds, "size": f"{size_x},{size_y}", "bboxSR": CRS, "imageSR": CRS, "f": "image", "format": "tiff", "adjustAspectRatio": "true", }, stream=True, headers=headers)

print(image_data.status_code)

if image_data.status_code == 200:
    with open(osp.join(data_dir, 'dem.tif'), 'wb') as f:
        image_data.raw.decode_content = True
        shutil.copyfileobj(image_data.raw, f)


# %%


dem = rd.LoadGDAL(osp.join(data_dir, 'dem.tif'), no_data=-9999)

rd.rdShow(dem, axes=True, cmap='Greys_r', figsize=(10, 6))


# ### Derive slope
# 

# %%


slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
rd.rdShow(slope, axes=True, cmap='magma', figsize=(8, 5.5))


# %%


rd.SaveGDAL(osp.join(data_dir, 'slope.tif'), slope/slope.max())


# ### Derive aspect
# 

# %%


aspect = rd.TerrainAttribute(dem, attrib='aspect')
rd.rdShow(aspect, axes=True, cmap='jet', figsize=(8, 5.5))


# %%


rd.SaveGDAL(osp.join(data_dir, 'aspect.tif'), aspect/360)


# ### Derive curvature
# 

# %%


curvature = rd.TerrainAttribute(dem, attrib='curvature')
rd.rdShow(curvature, axes=True, cmap='jet_r', figsize=(8, 5.5))


# %%


c_norm = (curvature + abs(curvature.min())) / curvature.max()
rd.SaveGDAL(osp.join(data_dir, 'curvature.tif'), c_norm)


# # Preprocess data
# 

# ## Rasterize shapefile data

# %%


# function to convert shapefile to a raster
def shp2raster(example_raster_path, shp_gdf, output_path, attr=None):

    # read in vector
    vector = shp_gdf.copy()

    # get list of geometries for all features in vector file
    geom = [shapes for shapes in vector.geometry]

    if attr is not None:
        # create a numeric unique value for each attribute/feature
        le = preprocessing.LabelEncoder()
        le.fit(vector[attr])
        vector[attr] = le.transform(vector[attr])
        vector[attr] = [x+1 for x in vector[attr]]

        # create tuples of geometry, value pairs, where value is the attribute value you want to burn
        geom = ((geom, value)
                for geom, value in zip(vector.geometry, vector[attr]))

    # open example raster
    raster = rasterio.open(example_raster_path)

    # rasterize vector using the shape and raster CRS
    rasterized = features.rasterize(geom,
                                    out_shape=raster.shape,
                                    fill=0,
                                    out=None,
                                    transform=raster.transform,
                                    all_touched=False,
                                    default_value=1,
                                    dtype=None)

    # rd.SaveGDAL(output_path, rasterized)

    kwargs = raster.meta
    kwargs.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw')

    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write_band(1, rasterized.astype(rasterio.float32))

    return rasterized


# %%





# %%


# convert the shapefiles to rasters
scarps_rast = shp2raster(osp.join(data_dir, 'dem.tif'),
                         scarps, output_path=osp.join(data_dir, 'scarps.tif'))

deposits_rast = shp2raster(osp.join(
    data_dir, 'dem.tif'), deposits, output_path=osp.join(data_dir, 'deposits.tif'))

lithology_rast = shp2raster(osp.join(data_dir, 'dem.tif'), lithology, output_path=osp.join(
    data_dir, 'lithology.tif'), attr='GENERAL_LITHOLOGY')


# ### Visualize rasterized data

# %%


# Plot scarps raster
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.set_title('Rasterized Landslide Scarps')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
ax.grid(True)
show(scarps_rast, ax=ax, cmap='Greys')


# %%


# Plot deposits raster
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.set_title('Rasterized Landslide Deposits')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
ax.grid(True)
show(deposits_rast, ax=ax, cmap='Greys')


# %%


# Plot lithology raster
fig, ax = plt.subplots(1, figsize=(8, 8))
ax.set_title('Rasterized Lithology')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
ax.grid(True)
show(lithology_rast, ax=ax)


# ## Create a composite raster of input data

# %%


# create a composite raster of the data
def composite(rasters, output_path):

    # open the first raster in the list
    with rasterio.open(rasters[0]) as rast:
        meta = rast.meta

    # update the metadata to reflect the composite raster # of layers
    meta.update(count=len(rasters))

    # read each raster layer and write it to the composite raster
    with rasterio.open(output_path, 'w', **meta) as dst:
        for id, raster in enumerate(rasters, start=1):
            with rasterio.open(raster) as rast:
                dst.writeband(id, rast.read(1))


# %%





# ### Check raster sizes

# %%


multi_bands = glob(data_dir + '/*.tif')
band_titles = [osp.basename(rast).split('.')[0] for rast in multi_bands]

# check to ensure the rasters have consistent sizes
for raster in multi_bands:
    r = gdal.Open(raster)
    print(osp.basename(raster).split('.')[0]+":", r.RasterYSize, r.RasterXSize)


# %%


stack, meta = es.stack([osp.join(data_dir, 'slope.tif'),
                        # osp.join(data_dir, 'slope.tif'),
                        # osp.join(data_dir, 'lithology.tif')],
                        osp.join(data_dir, 'curvature.tif'),
                       osp.join(data_dir, 'aspect.tif')],
                       osp.join(data_dir, 'composite.tif'))


# %%


plt.imshow(stack[0, :, :], cmap='magma')


# ## Data augmentation
# ### Tile the rasters

# %%


# load rasters

# # labels
# _, scarp_labels = pyrsgis.raster.read(
#     osp.join(data_dir, 'scarps.tif'), bands=1)

# _, deposit_labels = pyrsgis.raster.read(
#     osp.join(data_dir, 'deposits.tif'), bands=1)

# # features
# _, comp_feat = pyrsgis.raster.read(
#     osp.join(data_dir, 'composite.tif'), bands='all')


# %%


scarp_labels = scarps_rast.astype(np.uint8)
deposit_labels = deposits_rast.astype(np.uint8)
# comp_feat = stack.astype(np.float32)
comp_feat = stack.astype(np.float32)


# %%


# convert all negatives to 0
# tmp = np.clip(scarp_labels, a_min=0, a_max=np.inf)
scarp_labels = np.clip(scarp_labels, a_min=0, a_max=np.inf) # np.dstack((tmp, scarp_labels))
print(scarp_labels.shape)

# tmp = np.clip(deposit_labels, a_min=0, a_max=np.inf)
deposit_labels = np.clip(deposit_labels, a_min=0, a_max=np.inf) # np.dstack((tmp, deposit_labels))
print(deposit_labels.shape)


# %%


# reshape features
comp_feat = comp_feat.reshape(*scarp_labels.shape[:2], -1)
print(comp_feat.shape)


# %%


x_split = size_x/TILE_SIZE
y_split = size_y/TILE_SIZE

X = np.concatenate([np.hsplit(col, x_split) for col in 
                   np.vsplit(comp_feat, y_split)])

y_scarps = np.concatenate([np.hsplit(col, x_split)
                           for col in np.vsplit(scarp_labels, y_split)])

y_deposits = np.concatenate([np.hsplit(col, x_split)
                             for col in np.vsplit(deposit_labels, y_split)])

print(X.shape, y_scarps.shape, y_deposits.shape)


# ### Augment the data and masks

# %%


y = np.expand_dims(y_scarps.astype(np.uint8), axis=-1) 

seq = iaa.Sequential([
    iaa.Fliplr(p=0.5, seed=seed),  # 50% chance to flip horizontally
    iaa.Flipud(p=0.5, seed=seed),  # 50% chance to flip vertically
    # iaa.Rot90((1, 3)),  # rotate by 90, 180 or 270 degrees
    # iaa.PiecewiseAffine(scale=(0.01, 0.05))
])

X_aug, y_aug = seq(images=X, segmentation_maps=y)

# combine the original and augmented images
X_scarps = np.vstack((X, X_aug))
y_scarps = np.vstack((y, y_aug))[..., 0]


# %%


#y = y_deposits.astype(np.uint8)  # convert masks to boolean
y = np.expand_dims(y_deposits.astype(np.uint8), axis=-1) 

X_aug, y_aug = seq(images=X, segmentation_maps=y)

# combine the original and augmented images
X_deposits = np.vstack((X, X_aug))
y_deposits = np.vstack((y, y_aug))[..., 0]

print(X_scarps.shape, y_scarps.shape, X_deposits.shape, y_deposits.shape)


# %%


# # normalize the data
X_scarps_norm = tf.image.per_image_standardization(X_scarps).numpy()
X_deposits_norm = tf.image.per_image_standardization(X_deposits).numpy()
# X_norm = tf.image.per_image_standardization(X).numpy()


# ## Visualize augmented data

# %%


for i in range(0, 3): # get three random samples
    n = np.random.randint(0, len(X))

    f = plt.figure(figsize=(10, 10)) # make a figure

    # plot the composite tile image
    ax1 = f.add_subplot(1, 4, 1)
    ax1.title.set_text(f'Image {n}')
    plt.imshow(X_scarps_norm[n])

    # plot the scarps mask
    ax2 = f.add_subplot(1, 4, 2)
    ax2.title.set_text(f'Scarps Mask {n}')
    plt.imshow(y_scarps[n], cmap='gray')

    # ax3 = f.add_subplot(1, 4, 3)
    # ax3.title.set_text(f'Deposits Image {n}')
    # plt.imshow(X_deposits_norm[n], cmap='gray')

    # plot the deposits mask
    ax4 = f.add_subplot(1, 4, 3)
    ax4.title.set_text(f'Deposits Mask {n}')
    plt.imshow(y_deposits[n], cmap='gray')

    plt.show(block=True) # show the figure


# # Model architecture

# %%


def conv_block(x, n_filters):

    x = layers.Conv2D(n_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(n_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    return x


# %%





# %%


def downsample_block(x, n_filters):

    f = conv_block(x, n_filters)
    p = layers.MaxPool2D((2, 2))(f)
    
    return f, p


# %%





# %%


def upsample_block(x, conv_features, n_filters):

    x = layers.Conv2DTranspose(n_filters, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, conv_features])
    x = conv_block(x, n_filters)
    
    return x


# %%





# ## Build the model

# %%


def build_unet_model(input_shape, n_classes):

    # inputs
    inputs = layers.Input(shape=input_shape)

    n = input_shape[0]/2

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, n)
    # 2 - downsample
    f2, p2 = downsample_block(p1, n*2)
    # 3 - downsample
    f3, p3 = downsample_block(p2, n*4)
    # 4 - downsample
    f4, p4 = downsample_block(p3, n*8)

    # 5 - bottleneck
    bottleneck = conv_block(p4, n*16)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, n*8)
    # 7 - upsample
    u7 = upsample_block(u6, f3, n*4)
    # 8 - upsample
    u8 = upsample_block(u7, f2, n*2)
    # 9 - upsample
    u9 = upsample_block(u8, f1, n)

    # outputs
    outputs = layers.Conv2D(n_classes, 1, padding="same",
                            activation="sigmoid")(u9)

    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="unet_model")

    return unet_model


# %%





# ## Compile the model

# %%


# define model parameters
TRAIN_SIZE = 0.7
VALID_SIZE = 0.15
TEST_SIZE = 0.15
BATCH_SIZE = 32
N_CLASSES = y.shape[-1]
INPUT_SHAPE = (TILE_SIZE, TILE_SIZE, X.shape[-1])


# %%





# %%


unet = build_unet_model(input_shape=INPUT_SHAPE, n_classes=N_CLASSES)
unet.summary()


# %%





# %%


tf.keras.utils.plot_model(unet, to_file=osp.join('..', 'refs', 'model.png'), show_shapes=True)


# %%





# %%


unet.compile(optimizer=tf.keras.optimizers.Adam(),
             loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
             metrics=['accuracy'])

# free up RAM in case the model definition cells were run multiple times
tf.keras.backend.clear_session()


# # Train model to identify deposits
# ## Define model inputs and parameters
# 
# 

# %%


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_deposits_norm, y_deposits, test_size=TEST_SIZE, shuffle=True, random_state=seed)


# ### Callbacks
# 

# %%


output_dir = f'../models/'
file_name = 'deposits_unet.hdf5'
file_path = osp.join(output_dir, file_name)


# %%


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    file_path, verbose=1, save_best_only=True, mode='min')

# keep track of the model training progression
history = tf.keras.callbacks.History()

callbacks = [checkpoint,
             tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(
                 log_dir=osp.join('..', 'deposits_logs')),
             history]


# ## Train the deposits model
# 

# %%


results = unet.fit(X_train, y_train, batch_size=32,
                   epochs=50,
                   validation_split=VALID_SIZE,
                   callbacks=callbacks,
                   use_multiprocessing=True,
                   workers=multiprocessing.cpu_count()-1,
                   verbose=2)


# ## Evaluate the deposits model on the test set
# 

# %%


best_model = tf.keras.models.load_model(file_path)

# use the model to make predictions on the reserved test data
score = best_model.evaluate(x=X_test, y=y_test, verbose=2)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# # Train the model to identify scarps
# ## Define model inputs and parameters
# 

# %%


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scarps_norm, y_scarps, test_size=TEST_SIZE, shuffle=True, random_state=seed)

# X_train, X_test, y_train, y_test = train_test_split(X_norm, y_scarps, test_size=TEST_SIZE, shuffle=True, random_state=seed)


# ### Callbacks
# 

# %%


output_dir = f'../models/'
file_name = 'scarps_unet.hdf5'
file_path = osp.join(output_dir, file_name)


# %%


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    file_path, verbose=2, save_best_only=True, mode='min')

# keep track of the model training progression
history = tf.keras.callbacks.History()

callbacks = [checkpoint,
             tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
             tf.keras.callbacks.TensorBoard(log_dir=osp.join('..', 'scarps_logs')),
             history]


# ## Train the scarps model
# 

# %%


results = unet.fit(X_train, y_train, batch_size=32,
                   epochs=50,
                   validation_split=VALID_SIZE,
                   callbacks=callbacks,
                   use_multiprocessing=True,
                   workers=multiprocessing.cpu_count()-1,
                   verbose=2)


# ## Evaluate the scarps model on the test set
# 

# %%


best_model = tf.keras.models.load_model(file_path)

# use the model to make predictions on the reserved test data
score = best_model.evaluate(x=X_test, y=y_test, verbose=2)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


# %%





# # Use the trained models to make predictions

# %%





# ## View the model training metrics online with TensorBoard
# 
# [Open Tensorboard](https://tensorboard.dev/experiment/mNL39anoRmKK0QWHeoVeaQ/#scalars)
