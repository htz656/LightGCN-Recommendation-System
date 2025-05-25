import torch

from managers.Manager import BaseManager, ManagerOption
from models.LightGCN import LightGCN


class LightGCNManager(BaseManager):
    def __init__(self, option: ManagerOption, out_message = None, out_console = None):
        super(LightGCNManager, self).__init__(option, out_message, out_console)

    def _init_model(self):
        if self.model is None:
            self._load_full_dataset()
            self.model = LightGCN(
                self.full_dataset.num_users,
                self.full_dataset.num_items,
                self.option.embed_dim,
                self.option.num_layers,
                self.norm_adj_tensor,
                self.option.dropout
            ).to(self.device)

    def _load_model(self):
        dummy_norm_adj = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            (1, 1)  # Dummy shape
        ).coalesce()

        self.model = LightGCN(
            len(self.user2id),
            len(self.item2id),
            self.option.embed_dim,
            self.option.num_layers,
            dummy_norm_adj,
            0.0
        )
