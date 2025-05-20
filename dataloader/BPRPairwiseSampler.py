import random
import torch
from torch.utils.data import Dataset

class BPRPairwiseSampler(Dataset):
    def __init__(self, interaction_matrix, num_negatives=1):
        """
        interaction_matrix: torch.Tensor, shape [num_users, num_items]
        num_negatives: 每个正样本对应的负样本数量（每对都会变成独立样本）
        """
        self.inter_mat = interaction_matrix
        self.num_users, self.num_items = interaction_matrix.shape
        self.user_pos_dict = self._build_user_pos_dict()
        self.num_negatives = num_negatives
        self.user_pos_pairs = self._build_user_pos_pairs()
        self.total_triplets = len(self.user_pos_pairs) * num_negatives

    def _build_user_pos_dict(self):
        user_pos_dict = {}
        user_item_indices = torch.nonzero(self.inter_mat > 0, as_tuple=False)
        for u, i in user_item_indices:
            user_pos_dict.setdefault(u.item(), set()).add(i.item())
        return user_pos_dict

    def _build_user_pos_pairs(self):
        pairs = []
        for user, pos_items in self.user_pos_dict.items():
            for pos_item in pos_items:
                pairs.append((user, pos_item))
        return pairs

    def _sample_neg(self, pos_items):
        """从非交互项目中随机采样一个负样本"""
        while True:
            neg_item = random.randint(0, self.num_items - 1)
            if neg_item not in pos_items:
                return neg_item

    def __len__(self):
        return self.total_triplets

    def __getitem__(self, idx):
        base_idx = idx // self.num_negatives
        user, pos_item = self.user_pos_pairs[base_idx]
        pos_items = self.user_pos_dict[user]
        neg_item = self._sample_neg(pos_items)
        return user, pos_item, neg_item
