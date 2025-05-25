import datetime
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import List

import torch
from torch.utils.data import DataLoader

from dataloader.BPRPairwiseSampler import BPRPairwiseSampler
from dataloader.GraphaDataset import GraphaTrainDataset, GraphaTestDataset
from dataloader.LastFMDataset import LastFMDataset
from dataloader.MovieLensDataset import MovieLensDataset
from utils.loss import l2_reg_loss, bpr_loss
from utils.metrics import ranking_evaluation


@dataclass
class ManagerOption(object):
    is_train: bool = False
    listen_events: bool = False
    stop_flag = None

    device: str = "cuda:0"
    model_name: str = None
    embed_dim: int = 32
    num_layers: int = 3
    topN: List[int] = field(default_factory=lambda: [5, 10, 20])

    # 训练参数
    dataset: str = None
    data_dir: str = None
    batch_size: int = 2048
    num_negatives: int = 1

    reg: float = 0.00001
    lr: float = 0.001
    dropout: float = 0.0

    epochs: int = 500
    eval_freq: int = 20
    save_path: str = None

    # 推断参数
    load_path: str = None
    users: list[int] = field(default_factory=lambda: [])

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class BaseManager(object):
    _FD_DICT = {
        "LastFM": LastFMDataset,
        "MovieLens": MovieLensDataset
    }

    def __init__(self, option: ManagerOption, out_message = None, out_console = None):
        self.option = option
        self.out_message = out_message
        self.out_console = out_console
        self.device = torch.device(option.device)
        self.is_train = option.is_train

        self.full_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.norm_adj_tensor = None

        self.model = None

        self.best_user_emb = None
        self.best_item_emb = None
        self.save_path = None

        self.user2id = None
        self.item2id = None
        self.id2user = None
        self.id2item = None
        self.item2iname = None

    def _load_full_dataset(self):
        if self.full_dataset is None:
            assert self.option.dataset in self._FD_DICT
            self.full_dataset = self._FD_DICT[self.option.dataset](self.option.data_dir)
            self.norm_adj_tensor = self.full_dataset.build_normalized_adj_matrix(
                self.full_dataset.get_train_data()
            )
            self.user2id, self.item2id, self.id2user, self.id2item, self.item2iname = (
                self.full_dataset.get_user_item_mappings())

    def _load_train_dataset(self):
        if self.train_dataset is None:
            self._load_full_dataset()
            self.train_dataset = GraphaTrainDataset(self.full_dataset)

    def _load_test_dataset(self):
        if self.test_dataset is None:
            self._load_full_dataset()
            self.test_dataset = GraphaTestDataset(self.full_dataset)

    def _init_model(self):
        raise NotImplementedError

    def evaluate(self, user_emb, item_emb):
        self._load_test_dataset()
        self.model.eval()

        test_data = self.test_dataset.test_data
        ground_truth = {}

        for _, row in test_data.iterrows():
            user = int(row['user_idx'])
            item = int(row['item_idx'])
            ground_truth.setdefault(user, {})[item] = 1

        scores = torch.matmul(user_emb, item_emb.T)
        if self.train_dataset:
            train_matrix = self.train_dataset.get_interaction_binary_matrix().to(scores.device)
            # 屏蔽训练集中已交互过的项
            scores = scores * (1 - (train_matrix > 0).float()) - 1e10 * (train_matrix > 0).float()

        _, top_items = torch.topk(scores, k=max(self.option.topN), dim=-1)
        top_items = top_items.cpu().numpy()

        # 只对 ground_truth 中出现的用户生成推荐结果
        result = {u: [(int(i), 1.0) for i in top_items[u]] for u in ground_truth}

        c_top_n = [max(self.option.topN)] if self.is_train else self.option.topN
        report = ranking_evaluation(ground_truth, result, c_top_n)

        for n in c_top_n:
            if self.out_message is None:
                print(f"Top {n}")
                print(f"Hit Ratio: {report[n]['Hit Ratio']}")
                print(f"Precision: {report[n]['Precision']}")
                print(f"Recall: {report[n]['Recall']}")
                print(f"NDCG: {report[n]['NDCG']}")
            else:
                self.out_message(self.out_console, f"Top {n}")
                self.out_message(self.out_console, f"Hit Ratio: {report[n]['Hit Ratio']}")
                self.out_message(self.out_console, f"Precision: {report[n]['Precision']}")
                self.out_message(self.out_console, f"Recall: {report[n]['Recall']}")
                self.out_message(self.out_console, f"NDCG: {report[n]['NDCG']}")

        return report, c_top_n

    def train(self):
        self._init_model()
        self._load_train_dataset()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.option.save_path,
                                 f"{self.option.model_name}_{self.option.dataset}_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path

        train_sampler = BPRPairwiseSampler(
            interaction_matrix=self.train_dataset.get_interaction_weight_matrix(),
            num_negatives=self.option.num_negatives
        )
        train_loader = DataLoader(
            train_sampler,
            batch_size=self.option.batch_size,
            shuffle=True,
            drop_last=True
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option.lr)

        self.is_train = True
        best_recall = float('-inf')
        for epoch in range(self.option.epochs):
            if self.option.listen_events and self.option.stop_flag.is_set():
                self.out_message(self.out_console, "操作被用户终止。")
                return

            self.model.train()
            total_loss = 0.0
            for users, pos_items, neg_items in train_loader:
                users, pos_items, neg_items = \
                    users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)
                user_emb, item_emb = self.model()
                u = user_emb[users]
                pos = item_emb[pos_items]
                neg = item_emb[neg_items]
                loss = bpr_loss(u, pos, neg) + l2_reg_loss(self.option.reg, u, pos, neg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if self.out_message is None:
                print(f"Epoch {epoch + 1}/{self.option.epochs}, Loss: {total_loss:.4f}")
            else:
                self.out_message(self.out_console, f"Epoch {epoch + 1}/{self.option.epochs}, Loss: {total_loss:.4f}")

            if (epoch + 1) % self.option.eval_freq == 0:
                with torch.no_grad():
                    user_emb, item_emb = self.model()
                report, c_top_n = self.evaluate(user_emb, item_emb)
                if report[c_top_n[0]]['Recall'] > best_recall:
                    best_recall = report[c_top_n[0]]['Recall']
                    self.best_user_emb = user_emb.clone()
                    self.best_item_emb = item_emb.clone()
                    self.save()
                    if self.out_message is None:
                        print(f"New best Recall@{c_top_n[0]}: {best_recall:.4f}")
                    else:
                        self.out_message(self.out_console, f"New best Recall@{c_top_n[0]}: {best_recall:.4f}")

    def test(self):
        print("\nTesting best model on test set...")
        self._init_model()
        self._load_test_dataset()
        self.is_train = False
        if self.best_user_emb is None or self.best_item_emb is None:
            if self.out_message is None:
                print("No user embeddings or item embeddings...")
            else:
                self.out_message(self.out_console, "No user embeddings or item embeddings...")
            return None
        self.evaluate(self.best_user_emb, self.best_item_emb)
        return None

    def predict(self):
        if self.model is None:
            if self.out_message is None:
                print("model is None")
            else:
                self.out_message(self.out_console, "model is None")
            try:
                self.load()
            except FileNotFoundError:
                if self.out_message is None:
                    print("model can't be loaded!")
                else:
                    self.out_message(self.out_console, "model can't be loaded!")

        if self.best_user_emb is None or self.best_item_emb is None:
            if self.out_message is None:
                print("No user embeddings or item embeddings...")
            else:
                self.out_message(self.out_console, "No user embeddings or item embeddings...")
            return None
        scores = torch.matmul(self.best_user_emb, self.best_item_emb.T)

        result_dict = {}

        for n in self.option.topN:
            _, top_items = torch.topk(scores, k=n, dim=-1)
            top_items_np = top_items.cpu().numpy()

            for user in self.option.users:
                if self.option.listen_events and self.option.stop_flag.is_set():
                    self.out_message(self.out_console, "操作被用户终止。")
                    break

                if user not in self.user2id:
                    if self.out_message is None:
                        print(f"user {user} not in user2id...")
                    else:
                        self.out_message(self.out_console, f"user {user} not in user2id...")
                    continue

                uid = self.user2id[user]
                item_indices = top_items_np[uid][:n]
                # inames = [
                #     item2iname.get(id2item[iid], "Unknown") for iid in item_indices
                # ]
                inames = []
                for iid in item_indices:
                    item = self.id2item[iid]
                    iname = self.item2iname[item]
                    inames.append(iname)

                if uid not in result_dict:
                    result_dict[uid] = {}
                result_dict[uid][n] = inames

        for key, value in result_dict.items():
            for key_i, value_i in value.items():
                message = f"user {key} top{key_i}:\n"
                for iname in value_i:
                    if self.option.listen_events and self.option.stop_flag.is_set():
                        self.out_message(self.out_console, "操作被用户终止。")
                        return None
                    message += f"\t{iname}\n"
                if self.out_message is None:
                    print(message)
                else:
                    self.out_message(self.out_console, message)

        return result_dict

    def save(self):
        if self.model is None:
            raise RuntimeError("模型尚未初始化或训练，无法保存。")
        os.makedirs(self.option.save_path, exist_ok=True)

        option_dict = {
            "model_name": self.option.model_name,
            "embed_dim": self.option.embed_dim,
            "num_layers": self.option.num_layers
        }

        save_dict = {
            "model_state": self.model.state_dict(),
            "best_user_emb": self.best_user_emb.cpu() if self.best_user_emb is not None else None,
            "best_item_emb": self.best_item_emb.cpu() if self.best_item_emb is not None else None,
            "option": option_dict,
            "user2id": self.user2id,
            "item2id": self.item2id,
            "id2user": self.id2user,
            "id2item": self.id2item,
            "item2iname": self.item2iname,
        }
        save_file = os.path.join(self.save_path, f"ckpt.pt")
        torch.save(save_dict, save_file)

    def _load_model(self):
        raise NotImplementedError

    def load(self):
        ckpt_path = os.path.join(self.option.load_path, "ckpt.pt")
        try:
            # 尝试默认加载
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        except pickle.UnpicklingError:
            if self.out_message is None:
                print("UnpicklingError caught: retrying with weights_only=False...")
            else:
                self.out_message(self.out_console, "UnpicklingError caught: retrying with weights_only=False...")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        saved_option = checkpoint.get("option", None)
        if saved_option is None:
            raise RuntimeError("加载的模型中缺少参数信息（args），无法校验参数一致性。")

        keys_to_check = ["model_name", "embed_dim", "num_layers"]
        for key in keys_to_check:
            saved_val = saved_option.get(key, None)
            current_val = getattr(self.option, key, None)
            if saved_val != current_val:
                raise RuntimeError(
                    f"参数不匹配: 当前 {key}={current_val} 与保存模型中的 {key}={saved_val} 不一致，"
                    "请确认使用正确的参数加载模型。"
                )

        self.best_user_emb = checkpoint.get("best_user_emb", None)
        self.best_item_emb = checkpoint.get("best_item_emb", None)

        self.user2id = checkpoint.get("user2id", None)
        self.item2id = checkpoint.get("item2id", None)
        self.id2user = checkpoint.get("id2user", None)
        self.id2item = checkpoint.get("id2item", None)
        self.item2iname = checkpoint.get("item2iname", None)

        self._load_model()

        model_state = checkpoint["model_state"]
        sparse_norm_adj = model_state.pop("norm_adj")
        self.model.load_state_dict(model_state, strict=False)
        self.model.to(self.device)
        self.model.norm_adj = sparse_norm_adj.to(self.device)
        self.model.eval()
