import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, n_layers, norm_adj):
        super(LightGCN, self).__init__()
        self.emb_size = embed_dim
        self.n_layers = n_layers
        self.num_users = num_users
        self.num_items = num_items

        # 用户嵌入矩阵，形状 [num_users, emb_size]
        self.user_emb = nn.Parameter(torch.randn(num_users, embed_dim))
        # 物品嵌入矩阵，形状 [num_items, emb_size]
        self.item_emb = nn.Parameter(torch.randn(num_items, embed_dim))
        # 归一化稀疏邻接矩阵，形状 [(num_users + num_items), (num_users + num_items)]
        self.norm_adj = norm_adj

    def forward(self):
        # 拼接用户和物品嵌入，形状 [num_users + num_items, emb_size]
        all_emb = torch.cat([self.user_emb, self.item_emb], dim=0)

        # embs 列表存放每层的嵌入，初始为第0层（原始嵌入）
        embs = [all_emb]  # 形状 [[num_users + num_items, emb_size], ]

        for _ in range(self.n_layers):
            # 图卷积传播，邻接矩阵乘以嵌入
            # all_emb: [num_users + num_items, emb_size]
            # norm_adj: sparse [(num_users + num_items), (num_users + num_items)]
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            # 传播后的all_emb仍然是 [num_users + num_items, emb_size]
            embs.append(all_emb)

        # 堆叠所有层嵌入，形状变成 [num_users + num_items, n_layers + 1, emb_size]
        embs = torch.stack(embs, dim=1)

        # 对所有层的嵌入取平均，形状变成 [num_users + num_items, emb_size]
        embs = torch.mean(embs, dim=1)

        # 分离用户和物品嵌入
        user_final = embs[:self.num_users]  # [num_users, emb_size]
        item_final = embs[self.num_users:]  # [num_items, emb_size]

        return user_final, item_final
