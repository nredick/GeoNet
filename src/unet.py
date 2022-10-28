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

# %%
# define a random seed for reproducibility
seed_picker = widgets.IntText(
    value=7,
    description='Seed:',
    disabled=False
)
seed_picker


# %%
seed = seed_picker.value

# set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed)

# set `python` built-in pseudo-random generator at a fixed value
random.seed(seed)

# set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed)

# set `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed)

# configure a new global `tensorflow` session
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
