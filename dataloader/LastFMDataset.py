import os

import pandas as pd

from dataloader.GraphaDataset import GraphaDataset


class LastFMLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_user_item_file(self):
        path = os.path.join(self.data_dir, 'user_artists.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['userID', 'artistID', 'weight']

    def load_iid_iname_file(self):
        path = os.path.join(self.data_dir, 'artists.dat')   # ['id', 'name', 'url', pictureURL]
        return pd.read_csv(path, sep='\t', engine='python', usecols = [0, 1], on_bad_lines = 'skip')


class LastFMDataset(GraphaDataset):
    USERID = 'userID'
    ITEMID = 'artistID'
    WEIGHT = 'weight'
    LOADER = LastFMLoader
    IID = 'id'
    INAME = 'name'

    def __init__(self, *kargs, **kwargs):
        super(LastFMDataset, self).__init__(*kargs, **kwargs)
        self._load_from(self.LOADER)
