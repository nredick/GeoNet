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

# %% [markdown]
# # **Geospatial Analysis with Machine Learning**
#
# # Imports & Environment Settings
#
# ## Imports
#

# %%
# imports and setup
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import random
import tensorflow as tf
from keras import backend as K

import ipywidgets as widgets
from keras import backend as K
import tensorflow as tf
from IPython.display import clear_output


# %% [markdown]
# ## Setting a random seed
#
# ### What is a random seed?
#
# A random seed is a number that is used to initialize a pseudorandom number generator. This is used to generate a sequence of numbers that are seemingly random, but are actually deterministic. This is useful for reproducibility, as the same seed will always generate the same sequence of numbers.
#
# In short, this allows the results of this workflow to be reproducible.
#
# Set the random seed to any integer value you please.
#

# %%
# define a random seed for reproducibility
seed_picker = widgets.IntText(
    value=42,
    description='Seed:',
    disabled=False
)
seed_picker



# %%
SEED = seed_picker.value

# set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(SEED)

# set `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)

# set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)

# set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(SEED)

# configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)


# %% [markdown]
# # Add Data
#
# ## Upload your data

# %%
add_data = widgets.FileUpload(
    accept='',  # todo: add file types 
    multiple=True  
)

add_data

# %% [markdown]
# ## Optional: Automatically add DEM data
#
# Automatically download and add DEM data to your project. This will add a new layer to your project called `dem`.
#
# This data is sourced from todo
#
# > This only works if your area of interest is in the USA

# %% [markdown]
# # Preliminary Data Analysis

# %%
# todo add pandas profiling with options for in depth not in depth, to save or not save to file

# %% [markdown]
# # Data Preparation
#
# Create your dataset!
#
# ## Select your tile size 
#
# ### What is a tile size?
#
# ### Why are tile sizes powers of 2?
#
# ### How to choose a tile size

# %%
select_tile_size = widgets.Dropdown(
    options=['16', '32', '64', '128', '256', '512'],
    value='128',
    description='Number:',
    disabled=False,
)
select_tile_size

# %%
TILE_SIZE = int(select_tile_size.value)

# %% [markdown]
# ## Rasterize data 

# %% [markdown]
# ## Build a composite raster

# %% [markdown]
# ## Crop to tiles 

# %% [markdown]
# ## Optional: Data Augmentation
#
# add a click to move to next section button/link
#
# > Defaults to `True`
#
# ### What is data augmentation?
#
# ### What kind of data augmentation should I use?

# %%
options = ['Rotation', 'Horizontal Flip', 'Vertical Flip', 'etc.']

data_aug_types = widgets.SelectMultiple(
    options=options,
    # value=['None'],
    rows=len(options),
    description='Select:',
    disabled=False
)

data_aug_types

# %%
if len(data_aug_types.value) == 0:
    print('No data augmentation methods selected.')
else: 
    for i in data_aug_types.value:
        print(i)
        # todo apply method 

# %% [markdown]
# # Data Visualization
#
# Check to make sure things look good visually

# %% [markdown]
# # Split Data into Training, Validation, & Test Sets
#
# ## Select training set size

# %%
train_size = widgets.FloatSlider(
    value=.80,
    min=0.01,
    max=1.0,
    step=0.01,
    description='Train:',
    disabled=False,
    continuous_update=True,
    # orientation='vertical',
    readout=True,
    readout_format='.2f',
)

train_size

# %%
TRAIN_SIZE = train_size.value
if TRAIN_SIZE < 0.50:
    print('Warning! Training size is less than 40%, this may result in poor model performance.')

# %% [markdown]
# ## Select validation set size

# %%
validation_size = widgets.FloatSlider(
    value=.10,
    min=0.00,
    max=1.1-TRAIN_SIZE,
    step=0.01,
    description='Validation:',
    disabled=False,
    continuous_update=True,
    # orientation='vertical',
    readout=True,
    readout_format='.2f',
)

validation_size

# %%
VALIDATION_SIZE = validation_size.value

# %% [markdown]
# ## Select test set size

# %%
test_size = widgets.FloatSlider(
    value=.10,
    min=0.00,
    max=1.1-TRAIN_SIZE-VALIDATION_SIZE,
    step=0.01,
    description='Test:',
    disabled=False,
    continuous_update=True,
    # orientation='vertical',
    readout=True,
    readout_format='.2f',
)

test_size

# %%
TRAIN_SIZE = train_size.value
VALIDATION_SIZE = validation_size.value
TEST_SIZE = test_size.value

try: 
    assert TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE == 1.0
except AssertionError:
    print('Error! Train, validation, and test sizes do not add up to 1.0. Please adjust the sliders and rerun this cell.')

# %%
# todo train test split 

# %% [markdown]
# # Train the Model
#
#

# %% [markdown]
# # Results
#
# ## Visualize Results

# %% [markdown]
# ## Calculate Performance Metrics

# %% [markdown]
# ## Interpret Results

# %% [markdown]
# # Export Model
#
# ## Optional: Generate Report

# %% [markdown]
# ## Optional: Create Tensorboard Report
