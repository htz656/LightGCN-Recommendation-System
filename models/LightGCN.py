import torch
import torch.nn as nn


class LightGCN(nn.Module):
    """
    LightGCN 模型实现

    该模型是一个轻量级的图卷积推荐模型，仅保留了邻居信息传播部分，去除了非线性激活函数和权重矩阵。
    """
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 embed_dim: int,
                 num_layers: int,
                 norm_adj: torch.Tensor):
        super(LightGCN, self).__init__()
        self.emb_size = embed_dim
        self.n_layers = num_layers
        self.num_users = num_users
        self.num_items = num_items

        # 初始化用户嵌入，形状为 [num_users, embed_dim]
        self.user_emb = nn.Parameter(torch.randn(num_users, embed_dim))
        # 初始化物品嵌入，形状为 [num_items, embed_dim]
        self.item_emb = nn.Parameter(torch.randn(num_items, embed_dim))

        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)

        # 保存归一化后的稀疏邻接矩阵，形状为 [(num_users + num_items), (num_users + num_items)]
        self.register_buffer('norm_adj', norm_adj)

    def forward(self):
        # 拼接用户和物品嵌入，得到 [num_users + num_items, embed_dim]
        all_emb = torch.cat([self.user_emb, self.item_emb], dim=0)

        # 用于存储每一层的嵌入，初始为第0层（即原始嵌入）
        embs = [all_emb]

        # 图卷积传播，执行 num_layers 次
        for layer in range(self.n_layers):
            # 执行一层图卷积传播
            # norm_adj 是稀疏邻接矩阵，乘以当前嵌入向量，传播信息
            all_emb = torch.sparse.mm(self.norm_adj, all_emb)
            # 每层传播结果追加到 embs 列表中
            embs.append(all_emb)

        # embs 是一个包含 n_layers+1 个元素的列表，每个元素形状为 [num_users + num_items, embed_dim]
        # 将其堆叠成一个三维张量，形状为 [num_users + num_items, n_layers + 1, embed_dim]
        embs = torch.stack(embs, dim=1)

        # 对 n_layers+1 个嵌入进行平均，得到最终嵌入
        # 平均后的形状为 [num_users + num_items, embed_dim]
        embs = torch.mean(embs, dim=1)

        # 拆分出最终的用户和物品嵌入
        user_final = embs[:self.num_users]  # [num_users, embed_dim]
        item_final = embs[self.num_users:]  # [num_items, embed_dim]

        return user_final, item_final
