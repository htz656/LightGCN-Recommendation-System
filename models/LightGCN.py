import torch
import torch.nn as nn

from utils.matrix import sparse_dropout


class LightGCN(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int,
                 num_layers: int,
                 norm_adj: torch.Tensor,
                 dropout: float = 0.0):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.register_buffer('norm_adj', norm_adj)

        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(torch.empty(num_users, embed_dim)),
            "item_emb": nn.Parameter(torch.empty(num_items, embed_dim))
        })
        nn.init.xavier_uniform_(self.embedding_dict["user_emb"])
        nn.init.xavier_uniform_(self.embedding_dict["item_emb"])

    def forward(self):
        if self.training and self.dropout > 0:
            norm_adj = sparse_dropout(self.norm_adj, self.dropout)
        else:
            norm_adj = self.norm_adj

        ego_embeddings = torch.cat([self.embedding_dict["user_emb"],
                                    self.embedding_dict["item_emb"]], dim=0)
        all_embeddings = [ego_embeddings]

        for _ in range(self.num_layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)

        user_final = all_embeddings[:self.num_users]
        item_final = all_embeddings[self.num_users:]
        return user_final, item_final
