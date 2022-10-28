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
# > Defaults to `True`
#
# ### What is data augmentation?
#
# ### What kind of data augmentation should I use?

# %%
chkboxes = widgets.Checkbox(
    value=False,
    description='Check me',
    disabled=False,
    indent=False
)
chkboxes
