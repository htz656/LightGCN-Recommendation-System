import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class LastFMLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_user_artists(self):
        path = os.path.join(self.data_dir, 'user_artists.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['userID', 'artistID', 'weight']

    def load_artists(self):
        path = os.path.join(self.data_dir, 'artists.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['id', 'name', 'url', 'pictureURL']

    def load_tags(self):
        path = os.path.join(self.data_dir, 'tags.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['tagID', 'tagValue']

    def load_user_tagged_artists(self):
        path = os.path.join(self.data_dir, 'user_taggedartists-timestamps.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['userID', 'artistID', 'tagID', 'timestamp']

    def load_user_friends(self):
        path = os.path.join(self.data_dir, 'user_friends.dat')
        return pd.read_csv(path, sep='\t', engine='python') # ['userID', 'friendID']


class LastFMDataset:
    def __init__(self, data_dir, test_size=0.2, seed=42):
        self.loader = LastFMLoader(data_dir)
        self.data = self.loader.load_user_artists()

        self.user2id = {uid: i for i, uid in enumerate(self.data['userID'].unique())}
        self.item2id = {aid: i for i, aid in enumerate(self.data['artistID'].unique())}
        self.id2user = {i: uid for uid, i in self.user2id.items()}
        self.id2item = {i: aid for aid, i in self.item2id.items()}

        self.data['user_idx'] = self.data['userID'].map(self.user2id)
        self.data['item_idx'] = self.data['artistID'].map(self.item2id)

        train_data, test_data = train_test_split(
            self.data[['user_idx', 'item_idx', 'weight']],
            test_size=test_size,
            random_state=seed
        )
        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)

        self.num_users = len(self.user2id)
        self.num_items = len(self.item2id)

    def build_interaction_matrix(self, data):
        mat = torch.zeros((self.num_users, self.num_items), dtype=torch.float32)
        for _, row in data.iterrows():
            mat[row['user_idx'], row['item_idx']] = row['weight']
        return mat

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_user_item_mappings(self):
        return self.user2id, self.item2id, self.id2user, self.id2item


class LastFMTrainDataset(Dataset):
    def __init__(self, full_dataset: LastFMDataset):
        self.train_data = full_dataset.get_train_data()
        self.interaction_matrix = full_dataset.build_interaction_matrix(self.train_data)
        self.num_users = full_dataset.num_users
        self.num_items = full_dataset.num_items

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        row = self.train_data.iloc[idx]
        return row['user_idx'], row['item_idx'], row['weight']

    def get_interaction_matrix(self):
        return self.interaction_matrix


class LastFMTestDataset(Dataset):
    def __init__(self, full_dataset: LastFMDataset):
        self.test_data = full_dataset.get_test_data()

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        row = self.test_data.iloc[idx]
        return row['user_idx'], row['item_idx'], row['weight']
