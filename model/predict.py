Loading
required
packages
import numpy as np

# importing torch packages
import torch
from torch.utils.data import Dataset


# This class to transform the data, so that it can be loaded to test Loader
class DatasetProcessing(Dataset):
    """
       This function is used to initialise the class variables - transform, data, target

       """

    def __init__(self, data, target,
                 transform=None):  # used to initialise the class variables - transform, data, target
        self.transform = transform
        self.data = data.reshape((-1, 120, 320)).astype(np.uint8)[:, :, :, None]
        self.target = torch.from_numpy(target).float()  # needs to be in torch.LongTensor dtype

    def __getitem__(self, index):  # used to retrieve the X and y index value and return it
        return self.transform(self.data[index]), self.target[index]

    def __len__(self):  # returns the length of the data
        return len(list(self.data))
