import os

import pandas as pd

from dataloader.Dataset import GraphaDataset


class MovieLensLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_user_item_file(self):
        path = os.path.join(self.data_dir, 'user_ratedmovies-timestamps.dat')
        return pd.read_csv(path, sep='\t', engine='python')  # ['userID', 'movieID', 'rating', 'timestamp']


class MovieLensDataset(GraphaDataset):
    USERID = 'userID'
    ITEMID = 'movieID'
    WEIGHT = 'rating'
    LOADER = MovieLensLoader

    def __init__(self, *kargs, **kwargs):
        super(MovieLensDataset, self).__init__(*kargs, **kwargs)
        self._load_from(self.LOADER)
