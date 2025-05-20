import torch
from torch.utils.data import DataLoader

from dataloader.BPRPairwiseSampler import BPRPairwiseSampler
from dataloader.LastFMDataset import LastFMDataset, LastFMTrainDataset, LastFMTestDataset
from utils.loss import l2_reg_loss, bpr_loss
from utils.metrics import ranking_evaluation
from models.LightGCN import LightGCN


class LightGCNManager:
    _MODEL_NAME = "LightGCN"
    _FD_DICT = {
        "LastFM": LastFMDataset,
    }
    _TRD_DICT = {
        "LastFM": LastFMTrainDataset,
    }
    _TED_DICT = {
        "LastFM": LastFMTestDataset,
    }

    def __init__(self, args):
        self.user_emb = None
        self.item_emb = None
        self.best_user_emb = None
        self.best_item_emb = None

        assert args.model_name == LightGCNManager._MODEL_NAME
        assert args.dataset in ['LastFM', 'MovieLens']
        assert args.data_dir is not None

        self.args = args

        # 训练和测试数据集划分
        self.full_dataset = LightGCNManager._FD_DICT[args.dataset](args.data_dir)
        self.train_dataset = LightGCNManager._TRD_DICT[args.dataset](self.full_dataset)
        self.test_dataset = LightGCNManager._TED_DICT[args.dataset](self.full_dataset)

        self.num_users = self.full_dataset.num_users
        self.num_items = self.full_dataset.num_items

        # 训练用负采样器（负采样逻辑在此采样器中实现）
        self.trainSampler = BPRPairwiseSampler(
            interaction_matrix=self.train_dataset.get_interaction_matrix(),
            num_negatives=self.args.num_negatives
        )

        self.train_loader = DataLoader(
            self.trainSampler,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.model = LightGCN(self.num_users,
                              self.num_items,
                              self.args.embed_dim,
                              self.args.num_layers,
                              self.train_dataset.get_interaction_matrix()
        ).cuda()

    def evaluate(self):
        self.model.eval()
        test_data = self.test_dataset.test_data
        ground_truth = {}

        for _, row in test_data.iterrows():
            user = int(row['user_idx'])
            item = int(row['item_idx'])
            ground_truth.setdefault(user, {})
            ground_truth[user][item] = 1  # test中所有的交互设为正例

        # 获取分数
        scores = torch.matmul(self.user_emb, self.item_emb.T)  # (n_users, n_items)

        # 排除训练集中的物品
        train_matrix = self.train_dataset.get_interaction_matrix().to(scores.device)
        scores = scores * (1 - (train_matrix > 0).float()) - 1e10 * (train_matrix > 0).float()

        _, top_items = torch.topk(scores, k=max(self.args.topN), dim=-1)
        top_items = top_items.cpu().numpy()

        result = {u: [(int(i), 1.0) for i in top_items[u]] for u in range(self.num_users)}
        report = ranking_evaluation(ground_truth, result, self.args.topN)

        for line in report:
            print(line, end='')

        # 返回用于 early stopping 的指标（例如 Recall@10）
        recall_str = [r for r in report if "Recall" in r and f"Top {self.args.topN[0]}" in report[report.index(r)-1]]
        if recall_str:
            recall = float(recall_str[0].split(':')[-1])
        else:
            recall = 0.0
        return recall

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            best_recall = float('-inf')

            for users, pos_items, neg_items in self.train_loader:
                users = users.cuda()
                pos_items = pos_items.cuda()
                neg_items = neg_items.cuda()

                user_emb, item_emb = self.model()
                u = user_emb[users]
                pos = item_emb[pos_items]
                neg = item_emb[neg_items]

                loss = bpr_loss(u, pos, neg) + l2_reg_loss(self.args.reg, u, pos, neg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.args.epochs}, Loss: {total_loss:.4f}")

            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()

            if (epoch + 1) % self.args.eval_freq == 0:
                recall = self.evaluate()
                # 保存最好结果
                if recall > best_recall:
                    best_recall = recall
                    self.best_user_emb = self.user_emb.clone()
                    self.best_item_emb = self.item_emb.clone()
                    print(f"New best metric: {best_recall:.4f}")

        # 训练完后恢复最优embedding
        if self.best_user_emb is not None:
            self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def test(self):
        print("\nTesting best model on test set...")
        if self.user_emb is None or self.item_emb is None:
            self.user_emb, self.item_emb = self.model()
        self.evaluate()

    def predict(self):
        # 可用于实际推荐任务，返回每个用户 top-N 物品
        scores = torch.matmul(self.user_emb, self.item_emb.T)
        top_scores, top_items = torch.topk(scores, k=max(self.args.topN), dim=-1)
        return top_items.cpu().numpy()

    def save(self):
        torch.save(self.model.state_dict(), f"{self.args.save_path}/{LightGCNManager._MODEL_NAME}.pt")

    def load(self):
        self.model.load_state_dict(torch.load(f"{self.args.save_path}/{LightGCNManager._MODEL_NAME}.pt"))
        self.model.eval()
