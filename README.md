# 基于 LightGCN 的图神经推荐系统

本项目实现了一个基于图神经网络的轻量级协同过滤推荐系统 —— 
[LightGCN](https://arxiv.org/abs/2002.02126)

---
## 数据集
支持 [HetRec 2011 LastFM-2K](https://grouplens.org/datasets/hetrec-2011/) 与 [MovieLens-2K](https://grouplens.org/datasets/hetrec-2011/)

---
## 项目结构

```
LightGCN-Recommender/
│
├── models/
│ ├── LightGCN.py           # 模型结构定义
│ └── LightGCNManager.py    # 训练、评估与模型管理器
├── datasetloader/
│ ├── Dataset.py            # 通用图数据集接口
│ ├── BPRPairwiseSampler.py # BPR 负采样器
│ ├── LastFMDataset.py      # LastFM 数据集
│ └── MovieLensDataset.py   # MovieLens 数据集
├── utils/
│ ├── loss.py
│ └── metrics.py
├── main.py
├── requirements.txt 
├── .gitignore
├── LICENSE
└── README.md
```

---
## 快速开始

```bash
python train.py \
    --model_name LightGCN \
    --dataset LastFM \
    --data_dir ./datasets/hetrec2011-lastfm-2k \
    --device cuda:0 \
    --epochs 500 \
    --batch_size 2048
```
| 参数名               | 默认值                               | 说明                        |
|-------------------|-----------------------------------|---------------------------|
| `--model_name`    | LightGCN                          | 模型名称                      |
| `--embed_dim`     | 32                                | 嵌入向量维度                    |
| `--num_layers`    | 3                                 | GCN 层数                    |
| `--dataset`       | LastFM                            | 使用的数据集：LastFM 或 MovieLens |
| `--data_dir`      | `./datasets/hetrec2011-lastfm-2k` | 数据路径                      |
| `--num_negatives` | 1                                 | 每个正样本的负样本数                |
| `--batch_size`    | 2048                              | 批量大小                      |
| `--lr`            | 1e-3                              | 学习率                       |
| `--reg`           | 1e-5                              | L2 正则化                    |
| `--eval_freq`     | 10                                | 每 N 个 epoch 评估一次          |
| `--topN`          | 5 10 20                           | Top-N 推荐结果评估              |
| `--save_path`     | ./ckpts                           | 模型保存路径                    |


---
## TODO
- 增加可视化模块（如 embedding 可视化）
- 支持多种 GNN 变种（如 NGCF、GraphSAGE）
- 引入 TensorBoard 训练监控
