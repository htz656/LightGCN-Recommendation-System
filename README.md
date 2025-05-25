# 基于图卷积神经的推荐系统

本项目实现了一个基于图神经网络的轻量级协同过滤推荐系统，可以使用LightGCN及NGCF实现离线训练以及推荐。


---
## 数据集
支持 [HetRec 2011 LastFM-2K](https://grouplens.org/datasets/hetrec-2011/) 与 [MovieLens-2K](https://grouplens.org/datasets/hetrec-2011/)

---
## 项目结构

```
LightGCN-Recommender/
│
├── models/
│ ├── NGCF.py               # NGCF模型结构定义
│ ├── LightGCN.py           # LightGCN模型结构定义
├── managers/
│ ├── Manager.py            # 定义基本管理类及方法，定义管理类参数
│ ├── NGCFManager.py        # NGCF训练、评估与模型管理器
│ └── LightGCNManager.py    # LightGCN训练、评估与模型管理器
├── datasetloader/
│ ├── GraphaDataset.py      # 通用图数据集接口
│ ├── BPRPairwiseSampler.py # BPR 负采样器
│ ├── LastFMDataset.py      # LastFM 数据集
│ └── MovieLensDataset.py   # MovieLens 数据集
├── utils/
│ ├── loss.py
│ ├── matrix.py
│ └── metrics.py
├── gui.py
├── main.py
├── requirements.txt 
├── .gitignore
├── LICENSE
└── README.md
```

---
## 快速开始

```bash
python main.py
```
去除了命令行启动，改为图像界面启动

---
## TODO
- 支持更多 GNN 变种
- 支持在线训练和推荐

