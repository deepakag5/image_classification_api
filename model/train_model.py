# %%-----------------------------------------------------------------------
# Loading required packages

import matplotlib.pyplot as plt
import numpy as np  # For storing data as numpy arrays
import pandas as pd
import timeit

# %%-----------------------------------------------------------------------

# importing torch packages
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image  # For handling the images
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset