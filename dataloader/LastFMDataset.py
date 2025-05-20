import os

import numpy as np
import pandas as pd
import scipy as sp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from dataloader.Dataset import GraphaDataset


class LastFMLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_user_item_file(self):
        path = os.path.join(self.data_dir, 'user_artists.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['userID', 'artistID', 'weight']


class LastFMDataset(GraphaDataset):
    USERID = 'userID'
    ITEMID = 'artistID'
    WEIGHT = 'weight'
    LOADER = LastFMLoader

    def __init__(self, *kargs, **kwargs):
        super(LastFMDataset, self).__init__(*kargs, **kwargs)
        self._load_from(self.LOADER)
