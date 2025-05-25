import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.matrix import sparse_dropout


class NGCF(nn.Module):
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int,
                 num_layers: int,
                 norm_adj: torch.Tensor,
                 dropout: float):
        super(NGCF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.register_buffer('norm_adj', norm_adj)
        self.dropout = dropout

        self.embedding_dict = nn.ParameterDict({
            "user_emb": nn.Parameter(torch.empty(num_users, embed_dim)),
            "item_emb": nn.Parameter(torch.empty(num_items, embed_dim))
        })
        nn.init.xavier_uniform_(self.embedding_dict["user_emb"])
        nn.init.xavier_uniform_(self.embedding_dict["item_emb"])

        self.weight_dict = nn.ParameterDict()
        for i in range(self.num_layers):
            self.weight_dict[f"W_gc_{i}"] = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.weight_dict[f"W_bi_{i}"] = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.weight_dict[f"b_gc_{i}"] = nn.Parameter(torch.zeros(embed_dim))
            self.weight_dict[f"b_bi_{i}"] = nn.Parameter(torch.zeros(embed_dim))
            nn.init.xavier_uniform_(self.weight_dict[f"W_gc_{i}"])
            nn.init.xavier_uniform_(self.weight_dict[f"W_bi_{i}"])

    def forward(self):
        if self.training and self.dropout > 0:
            norm_adj = sparse_dropout(self.norm_adj, self.dropout)
        else:
            norm_adj = self.norm_adj

        ego_embeddings = torch.cat([self.embedding_dict["user_emb"], self.embedding_dict["item_emb"]], dim=0)
        all_embeddings = [ego_embeddings]

        for k in range(self.num_layers):
            side_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            sum_embeddings = (torch.matmul(side_embeddings, self.weight_dict[f"W_gc_{k}"])
                              + self.weight_dict[f"b_gc_{k}"])

            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = (torch.matmul(bi_embeddings, self.weight_dict[f"W_bi_{k}"])
                             + self.weight_dict[f"b_bi_{k}"])

            ego_embeddings = F.leaky_relu(sum_embeddings + bi_embeddings, negative_slope=0.2)
            ego_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            all_embeddings.append(ego_embeddings)

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)

        user_final = all_embeddings[:self.num_users]
        item_final = all_embeddings[self.num_users:]
        return user_final, item_final
