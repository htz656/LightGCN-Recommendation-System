import numpy as np
import pandas as pd
import scipy as sp
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class GraphaDataset(object):
    USERID = None
    ITEMID = None
    WEIGHT = None
    IID = None
    INAME = None

    def __init__(self, data_dir, test_size=0.2, seed=42):
        self.data_dir = data_dir
        self.test_size = test_size
        self.seed = seed

        self.data = None

        self.num_users = None
        self.num_items = None

        self.user2id = None
        self.item2id = None
        self.id2user = None
        self.id2item  = None

        self.train_data = None
        self.test_data = None

    def _load_from(self, loader_class):
        self.loader = loader_class(self.data_dir)
        self.data = self.loader.load_user_item_file()
        self.item_iname_df = self.loader.load_iid_iname_file()
        self.item2iname = dict(zip(self.item_iname_df[self.IID].astype(int), self.item_iname_df[self.INAME]))

        self.user2id = {uid: i for i, uid in enumerate(self.data[self.USERID].unique())}
        self.item2id = {aid: i for i, aid in enumerate(self.data[self.ITEMID].unique())}
        self.id2user = {i: uid for uid, i in self.user2id.items()}
        self.id2item = {i: aid for aid, i in self.item2id.items()}
        self.num_users = len(self.user2id)
        self.num_items = len(self.item2id)

        self.data['user_idx'] = self.data[self.USERID].map(self.user2id)
        self.data['item_idx'] = self.data[self.ITEMID].map(self.item2id)

        train_list, test_list = [], []
        for user in self.data['user_idx'].unique():
            user_data = self.data[self.data['user_idx'] == user]
            if len(user_data) < 2:
                train_list.append(user_data)
            else:
                train_u, test_u = train_test_split(user_data, test_size=self.test_size, random_state=self.seed)
                train_list.append(train_u)
                test_list.append(test_u)

        self.train_data = pd.concat(train_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        if len(test_list) > 0:
            self.test_data = pd.concat(test_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        else:
            self.test_data = pd.DataFrame(columns=self.data.columns)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_user_item_mappings(self):
        return self.user2id, self.item2id, self.id2user, self.id2item, self.item2iname

    def build_interaction_weight_matrix(self, data):
        user_idx = data['user_idx'].astype(int).values
        item_idx = data['item_idx'].astype(int).values
        weights = data[self.WEIGHT].astype(np.float32).values

        mat = torch.zeros((self.num_users, self.num_items), dtype=torch.float32)
        mat[user_idx, item_idx] = torch.from_numpy(weights)
        return mat

    def build_interaction_binary_matrix(self, data):
        user_idx = data['user_idx'].astype(int).values
        item_idx = data['item_idx'].astype(int).values

        mat = torch.zeros((self.num_users, self.num_items), dtype=torch.float32)
        mat[user_idx, item_idx] = 1.0
        return mat

    def build_normalized_adj_matrix(self, data):
        num_nodes = self.num_users + self.num_items

        # 构建稀疏邻接矩阵 COO 格式，用户节点为 [0, num_users)，物品节点为 [num_users, num_users+num_items)
        row = []
        col = []
        data_val = []

        for _, row_data in data.iterrows():
            u = row_data['user_idx']
            i = row_data['item_idx'] + self.num_users  # 物品索引偏移
            w = row_data[self.WEIGHT]

            # 添加两个方向（对称邻接）
            row += [u, i]
            col += [i, u]
            data_val += [w, w]

        # 创建稀疏邻接矩阵 A（用户-物品二部图）
        adj = sp.sparse.coo_matrix((data_val, (row, col)), shape=(num_nodes, num_nodes))

        # 计算度向量并避免除以 0
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1  # 防止除以 0
        deg_inv_sqrt = np.power(deg, -0.5)
        deg_inv_sqrt[deg == 1] = 0  # 将原始为 0 的项设为 0

        # 构造 D^{-1/2} 对角矩阵
        d_inv_sqrt = sp.sparse.diags(deg_inv_sqrt)

        # 归一化邻接矩阵： D^{-1/2} * A * D^{-1/2}
        norm_adj = d_inv_sqrt.dot(adj).dot(d_inv_sqrt).tocoo()

        # 转换为 PyTorch 稀疏张量
        indices = torch.from_numpy(
            np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64)
        )
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = norm_adj.shape

        norm_adj_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(shape))
        return norm_adj_tensor.coalesce()


class GraphaTrainDataset(Dataset):
    def __init__(self, full_dataset: GraphaDataset):
        self.full_dataset = full_dataset

        self.train_data = full_dataset.get_train_data()
        self.num_users = full_dataset.num_users
        self.num_items = full_dataset.num_items

        self.interaction_weight_matrix = full_dataset.build_interaction_weight_matrix(self.train_data)
        self.interaction_binary_matrix = full_dataset.build_interaction_binary_matrix(self.train_data)
        self.norm_adj = full_dataset.build_normalized_adj_matrix(self.train_data)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        row = self.train_data.iloc[idx]
        return row['user_idx'], row['item_idx'], row[self.full_dataset.WEIGHT]

    def get_interaction_binary_matrix(self):
        return self.interaction_binary_matrix

    def get_interaction_weight_matrix(self):
        return self.interaction_weight_matrix

    def get_norm_adj(self):
        return self.norm_adj


class GraphaTestDataset(Dataset):
    def __init__(self, full_dataset: GraphaDataset):
        self.full_dataset = full_dataset
        self.test_data = full_dataset.get_test_data()

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        row = self.test_data.iloc[idx]
        return row['user_idx'], row['item_idx'], row[self.full_dataset.WEIGHT]
