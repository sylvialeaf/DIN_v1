# 独立 DIN 研究项目

一个**完全独立于 RecBole 框架**的序列推荐模型实现。

## 📌 项目背景

在使用 RecBole 框架时遇到了多个兼容性问题（NumPy 2.0、Pandas 版本、数据集格式等），因此决定从零开始实现一个轻量、稳定、可控的版本。

### 为什么选择自己实现？

| 方面 | RecBole | 本项目 |
|------|---------|--------|
| 依赖 | 复杂，容易版本冲突 | 简单，仅需 PyTorch + 常用库 |
| 调试 | 黑盒，难以定位问题 | 透明，全部代码可见 |
| 面试 | 难以解释内部实现 | 完全理解，可深入讲解 |
| 修改 | 需要了解框架架构 | 直接修改，即改即用 |
| 稳定性 | 受框架更新影响 | 自己控制，长期稳定 |

## 🚀 快速开始

### 环境要求

```bash
pip install torch numpy pandas matplotlib scikit-learn lightgbm tqdm
```

### 运行实验

```bash
cd standalone_din

# 运行所有实验
python run_experiments.py

# 或单独运行
python experiment1.py   # 序列长度 + 模型对比实验
python experiment2.py   # 方法对比 + 混合精排实验
python experiment3.py   # 消融实验
```

## 📁 项目结构

```
standalone_din/
├── models.py              # 所有模型定义 ⭐
│   ├── AttentionLayer     # DIN 核心注意力层
│   ├── DINRich            # 丰富特征 DIN
│   ├── DINRichLite        # 轻量版 DIN
│   ├── GRU4Rec            # GRU 序列推荐 (NEW)
│   ├── SASRec             # Transformer 序列推荐 (NEW)
│   ├── NARM               # 神经注意力推荐 (NEW)
│   └── SimpleAveragePoolingRich  # 平均池化基线
├── data_loader.py         # 数据加载器
│   ├── RichFeatureDataset # 丰富特征数据集
│   └── get_rich_dataloaders() # 支持 ml-100k/ml-1m
├── feature_engineering.py # 特征工程模块 ⭐
│   ├── FeatureProcessor   # 用户/物品特征处理
│   ├── InteractionFeatureExtractor # 交互特征提取
│   └── prepare_lightgbm_features() # LightGBM 特征准备
├── trainer.py             # 训练器
│   └── RichTrainer        # 支持丰富特征的训练器
├── hybrid_ranker.py       # 混合精排模块 ⭐ (创新点)
│   ├── DINEmbeddingExtractor # DIN 嵌入提取器
│   └── HybridRanker       # DIN + LightGBM 混合精排
├── experiment1.py         # 实验一: 序列长度 + 模型对比
├── experiment2.py         # 实验二: 方法对比 + 混合精排
├── experiment3.py         # 实验三: 消融实验
├── run_experiments.py     # 主入口
└── README.md              # 本文件
```

## 🔬 三个实验

### 实验一: 序列长度敏感性 + 模型对比

**研究问题**: 历史行为长度如何影响不同序列推荐模型？

**对比模型**:
- **DIN**: Deep Interest Network（注意力机制）
- **GRU4Rec**: 基于 GRU 的序列推荐
- **SASRec**: 基于 Transformer 的自注意力推荐
- **NARM**: 神经注意力推荐机器（GRU + 注意力）
- **AvgPool**: 平均池化基线

**测试序列长度**: 20, 50, 100, 150

### 实验二: DIN vs 传统方法 + 混合精排

**研究问题**: DIN 相比传统方法有多大优势？混合精排能否进一步提升？

**对比方法**:
- DIN: 深度兴趣网络
- GRU4Rec: GRU 序列模型
- AvgPool: 平均池化
- LightGBM: 特征工程 + 树模型
- **Hybrid**: DIN + LightGBM 混合精排 🆕

**创新点 - 混合精排**:
```
用户历史序列 → DIN → 用户兴趣向量 ─┐
                                    ├─→ LightGBM → 最终排序
用户/物品/时间特征 ─────────────────┘
```
- DIN 提取深度语义特征
- LightGBM 处理交叉特征和显式特征
- 结合两者优势

### 实验三: DIN 改进消融实验

**研究问题**: DIN 各组件的贡献如何？

**消融配置**:
- DIN-Full: 完整模型
- DIN-NoAttn: 去除注意力（使用平均池化）
- DIN-NoUserFeat: 去除用户特征
- DIN-NoItemFeat: 去除物品特征
- DIN-NoTimeFeat: 去除时间特征

## ⭐ 特征工程

### 用户特征
| 特征 | 描述 | 处理方式 |
|------|------|----------|
| age_bucket | 年龄分桶 | 7 桶 |
| gender | 性别 | M/F 编码 |
| occupation | 职业 | 21 类编码 |
| user_activity | 活跃度 | 交互数分桶 |

### 物品特征
| 特征 | 描述 | 处理方式 |
|------|------|----------|
| primary_genre | 主类型 | 19 类编码 |
| year_bucket | 发布年代 | 7 桶 |
| item_popularity | 热度 | 交互数分桶 |

### 序列特征
| 特征 | 描述 |
|------|------|
| history_genres | 历史物品类型序列 |
| history_years | 历史物品年代序列 |

### 时间特征
| 特征 | 描述 |
|------|------|
| time_hour | 小时分桶 (7 桶) |
| time_dow | 星期几 (1-7) |
| time_weekend | 是否周末 |

## 📊 支持的数据集

- **ml-100k**: MovieLens 100K (10万交互，943用户，1682物品)
- **ml-1m**: MovieLens 1M (100万交互，6040用户，3883物品)

数据集会自动下载到 `./data/` 目录。

## 🏃 预期运行时间

| 实验 | ml-100k | ml-1m |
|------|---------|-------|
| 实验一 | ~10 分钟 | ~1 小时 |
| 实验二 | ~8 分钟 | ~45 分钟 |
| 实验三 | ~5 分钟 | ~30 分钟 |

## 📈 输出结果

每个实验会在 `results/` 目录生成:
- `experiment*_results.csv`: 详细数值结果
- `experiment*_plot.png`: 可视化图表
- `experiment*_report.json`: 结构化报告

## 🔧 自定义

### 修改数据集
```python
# 在 experiment*.py 中修改
train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
    data_dir='./data',
    dataset_name='ml-1m',  # 改为 ml-1m
    max_seq_length=50,
    batch_size=256
)
```

### 添加新模型
1. 在 `models.py` 中定义模型类
2. 确保 `forward()` 方法接收 `batch` 字典
3. 在实验脚本中添加模型配置

## 📚 参考论文

- [DIN] Deep Interest Network for Click-Through Rate Prediction (Zhou et al., 2018)
- [GRU4Rec] Session-based Recommendations with Recurrent Neural Networks (Hidasi et al., 2016)
- [SASRec] Self-Attentive Sequential Recommendation (Kang & McAuley, 2018)
- [NARM] Neural Attentive Session-based Recommendation (Li et al., 2017)
