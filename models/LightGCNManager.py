import os

import torch
from torch.utils.data import DataLoader

from dataloader.BPRPairwiseSampler import BPRPairwiseSampler
from dataloader.Dataset import GraphaTrainDataset, GraphaTestDataset
from dataloader.LastFMDataset import LastFMDataset
from dataloader.MovieLensDataset import MovieLensDataset
from utils.loss import l2_reg_loss, bpr_loss
from utils.metrics import ranking_evaluation
from models.LightGCN import LightGCN


class LightGCNManager:
    _MODEL_NAME = "LightGCN"
    _FD_DICT = {
        "LastFM": LastFMDataset,
        "MovieLens": MovieLensDataset
    }

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        self.is_train = False

        self.full_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.norm_adj_tensor = None
        self.model = None

        self.user_emb = None
        self.item_emb = None
        self.best_user_emb = None
        self.best_item_emb = None

    def _load_full_dataset(self):
        if self.full_dataset is None:
            assert self.args.dataset in self._FD_DICT
            self.full_dataset = self._FD_DICT[self.args.dataset](self.args.data_dir)
            self.norm_adj_tensor = self.full_dataset.build_normalized_adj_matrix(
                self.full_dataset.get_train_data()
            )

    def _init_model(self):
        if self.model is None:
            self._load_full_dataset()
            self.model = LightGCN(
                self.full_dataset.num_users,
                self.full_dataset.num_items,
                self.args.embed_dim,
                self.args.num_layers,
                self.norm_adj_tensor
            ).to(self.device)

    def _load_train_dataset(self):
        if self.train_dataset is None:
            self._load_full_dataset()
            self.train_dataset = GraphaTrainDataset(self.full_dataset)

    def _load_test_dataset(self):
        if self.test_dataset is None:
            self._load_full_dataset()
            self.test_dataset = GraphaTestDataset(self.full_dataset)

    def train(self):
        self._init_model()
        self._load_train_dataset()

        train_sampler = BPRPairwiseSampler(
            interaction_matrix=self.train_dataset.get_interaction_weight_matrix(),
            num_negatives=self.args.num_negatives
        )
        train_loader = DataLoader(
            train_sampler,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.is_train = True
        best_recall = float('-inf')
        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0.0
            for users, pos_items, neg_items in train_loader:
                users, pos_items, neg_items = users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
                user_emb, item_emb = self.model()
                u = user_emb[users]
                pos = item_emb[pos_items]
                neg = item_emb[neg_items]
                loss = bpr_loss(u, pos, neg) + l2_reg_loss(self.args.reg, u, pos, neg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {total_loss:.4f}")

            if (epoch + 1) % self.args.eval_freq == 0:
                with torch.no_grad():
                    self.user_emb, self.item_emb = self.model()
                report, c_top_n = self.evaluate()
                if report[c_top_n[0]]['Recall'] > best_recall:
                    best_recall = report[c_top_n[0]]['Recall']
                    self.best_user_emb = self.user_emb.clone()
                    self.best_item_emb = self.item_emb.clone()
                    print(f"New best Recall@{c_top_n[0]}: {best_recall:.4f}")

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def evaluate(self):
        self._init_model()
        self._load_test_dataset()
        self.model.eval()

        test_data = self.test_dataset.test_data
        ground_truth = {}

        for _, row in test_data.iterrows():
            user = int(row['user_idx'])
            item = int(row['item_idx'])
            ground_truth.setdefault(user, {})[item] = 1

        scores = torch.matmul(self.user_emb, self.item_emb.T)
        if self.train_dataset:
            train_matrix = self.train_dataset.get_interaction_binary_matrix().to(scores.device)
            # 屏蔽训练集中已交互过的项
            scores = scores * (1 - (train_matrix > 0).float()) - 1e10 * (train_matrix > 0).float()

        _, top_items = torch.topk(scores, k=max(self.args.topN), dim=-1)
        top_items = top_items.cpu().numpy()

        # 只对 ground_truth 中出现的用户生成推荐结果
        result = {u: [(int(i), 1.0) for i in top_items[u]] for u in ground_truth}

        c_top_n = [max(self.args.topN)] if self.is_train else self.args.topN
        report = ranking_evaluation(ground_truth, result, c_top_n)

        for n in c_top_n:
            print(f"Top {n}")
            print(f"Hit Ratio: {report[n]['Hit Ratio']}")
            print(f"Precision: {report[n]['Precision']}")
            print(f"Recall: {report[n]['Recall']}")
            print(f"NDCG: {report[n]['NDCG']}")

        return report, c_top_n

    def test(self):
        print("\nTesting best model on test set...")
        self._init_model()
        self._load_test_dataset()
        if self.user_emb is None or self.item_emb is None:
            print("No user embeddings or item embeddings...")
            return None
        self.is_train = False
        return self.evaluate()

    def predict(self):
        self._init_model()
        if self.user_emb is None or self.item_emb is None:
            print("No user embeddings or item embeddings...")
            return None
        scores = torch.matmul(self.user_emb, self.item_emb.T)
        _, top_items = torch.topk(scores, k=max(self.args.topN), dim=-1)
        return top_items.cpu().numpy()

    def save(self):
        if self.model is None:
            raise RuntimeError("模型尚未初始化或训练，无法保存。")
        os.makedirs(self.args.save_path, exist_ok=True)
        save_dict = {
            "model_state": self.model.state_dict(),
            "best_user_emb": self.best_user_emb.cpu() if self.best_user_emb is not None else None,
            "best_item_emb": self.best_item_emb.cpu() if self.best_item_emb is not None else None,
            "args": vars(self.args),
        }
        torch.save(save_dict, f"{self.args.save_path}/{self._MODEL_NAME}.pt")

    def load(self):
        checkpoint = torch.load(f"{self.args.save_path}/{self._MODEL_NAME}.pt", map_location=self.device)
        saved_args = checkpoint.get("args", None)
        if saved_args is None:
            raise RuntimeError("加载的模型中缺少参数信息（args），无法校验参数一致性。")

        keys_to_check = ["embed_dim", "num_layers", "dataset", "model_name"]
        for key in keys_to_check:
            saved_val = saved_args.get(key, None)
            current_val = getattr(self.args, key, None)
            if saved_val != current_val:
                raise RuntimeError(
                    f"参数不匹配: 当前 {key}={current_val} 与保存模型中的 {key}={saved_val} 不一致，"
                    "请确认使用正确的参数加载模型。"
                )

        self._init_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.best_user_emb = checkpoint.get("best_user_emb", None)
        self.best_item_emb = checkpoint.get("best_item_emb", None)

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb
