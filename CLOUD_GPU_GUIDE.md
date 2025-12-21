# 🚀 云端 GPU 部署指南

本项目支持在多种云端 GPU 平台运行，以下是详细的部署步骤。

---

## 📋 目录
- [平台选择](#平台选择)
- [AutoDL 部署（推荐）](#autodl-部署)
- [Google Colab 部署](#google-colab-部署)
- [阿里云/腾讯云部署](#阿里云腾讯云部署)
- [运行实验](#运行实验)
- [预估时间和成本](#预估时间和成本)

---

## 🎯 平台选择

| 平台 | 成本 | GPU | 推荐度 | 适合场景 |
|------|------|-----|--------|----------|
| **AutoDL** | ¥1-2/小时 | RTX 3090/A5000 | ⭐⭐⭐⭐⭐ | 国内首选 |
| **Colab Pro** | $10/月 | T4/V100 | ⭐⭐⭐⭐ | 免费试用 |
| **阿里云 PAI-DSW** | ¥3-5/小时 | V100/A10 | ⭐⭐⭐ | 企业用户 |
| **腾讯云 GPU** | ¥2-4/小时 | T4/V100 | ⭐⭐⭐ | 按需使用 |

---

## 🔧 AutoDL 部署

### 步骤 1：注册和创建实例

1. 访问 [AutoDL](https://www.autodl.com)
2. 注册账号并充值（建议 ¥20-50 足够）
3. 创建实例：
   - 选择 GPU：**RTX 3090** 或 **A5000**（约 ¥1.5/小时）
   - 镜像：**PyTorch 2.0.0 + Python 3.10**
   - 数据盘：10GB 足够

### 步骤 2：上传项目

**方法一：通过 Git（推荐）**
```bash
# 在 AutoDL 终端执行
cd /root/autodl-tmp
git clone https://your-git-repo-url.git
# 或者
git clone https://github.com/yourusername/standalone_din.git
```

**方法二：通过 SCP/SFTP**
```bash
# 在本地执行（Windows PowerShell）
scp -r D:\aProject\Project_RecBole\RecBole1\standalone_din root@your-autodl-ip:/root/autodl-tmp/
```

**方法三：网盘/压缩包**
1. 将 `standalone_din` 文件夹压缩为 zip
2. 上传到网盘获取下载链接
3. 在 AutoDL 终端：
```bash
cd /root/autodl-tmp
wget "网盘链接" -O project.zip
unzip project.zip
```

### 步骤 3：安装依赖

```bash
cd /root/autodl-tmp/standalone_din
pip install torch numpy pandas matplotlib scikit-learn lightgbm tqdm tensorboard
```

### 步骤 4：运行实验

```bash
# 完整实验（两个数据集，约 2 小时）
python run_all_gpu.py

# 只跑 ml-100k（约 20 分钟）
python run_all_gpu.py --dataset ml-100k

# 只跑 ml-1m（约 90 分钟）
python run_all_gpu.py --dataset ml-1m

# 快速测试模式（验证环境，约 5 分钟）
python run_all_gpu.py --quick

# 指定 epoch 数量
python run_all_gpu.py --epochs 100
```

---

## 📓 Google Colab 部署

### 步骤 1：创建新笔记本

1. 访问 [Google Colab](https://colab.research.google.com)
2. 新建笔记本
3. 菜单：运行时 → 更改运行时类型 → GPU

### 步骤 2：上传项目

```python
# Cell 1: 挂载 Google Drive（可选）
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: 上传压缩包
from google.colab import files
uploaded = files.upload()  # 选择 standalone_din.zip

# Cell 3: 解压
!unzip standalone_din.zip -d /content/
```

### 步骤 3：安装依赖

```python
# Cell 4
!pip install lightgbm
```

### 步骤 4：运行实验

```python
# Cell 5
%cd /content/standalone_din
!python run_all_gpu.py --quick  # 先快速测试

# Cell 6: 完整实验
!python run_all_gpu.py
```

### 步骤 5：下载结果

```python
# Cell 7
from google.colab import files
import zipfile
import os

# 压缩结果
with zipfile.ZipFile('results.zip', 'w') as z:
    for f in os.listdir('results_gpu'):
        z.write(os.path.join('results_gpu', f))

files.download('results.zip')
```

---

## ☁️ 阿里云/腾讯云部署

### 阿里云 PAI-DSW

1. 登录阿里云控制台 → 人工智能平台 PAI
2. 创建 DSW 实例：
   - GPU 类型：V100 或 A10
   - 镜像：PyTorch 2.0
3. 打开 Terminal，按 AutoDL 步骤操作

### 腾讯云 GPU 云服务器

1. 购买 GPU 实例（按量计费）
2. 选择带 CUDA 的镜像
3. SSH 连接后按 AutoDL 步骤操作

---

## ⚡ 运行实验

### 命令参数说明

```bash
python run_all_gpu.py [参数]

参数:
  --dataset {ml-100k,ml-1m,both}  # 选择数据集，默认 both
  --quick                          # 快速测试模式
  --epochs N                       # 训练轮数，默认 50
```

### 推荐运行顺序

```bash
# 1. 验证环境
python run_all_gpu.py --quick

# 2. 测试小数据集
python run_all_gpu.py --dataset ml-100k --epochs 30

# 3. 完整实验
python run_all_gpu.py --epochs 50
```

---

## ⏱️ 预估时间和成本

### GPU 运行时间

| 配置 | ml-100k | ml-1m | 总计 |
|------|---------|-------|------|
| epochs=20, quick | 5 分钟 | 15 分钟 | 20 分钟 |
| epochs=50 | 15 分钟 | 60 分钟 | 1.5 小时 |
| epochs=100 | 30 分钟 | 120 分钟 | 2.5 小时 |

### 成本估算（AutoDL RTX 3090）

| 配置 | 时间 | 成本 |
|------|------|------|
| Quick 模式 | 20 分钟 | ¥0.5 |
| 标准模式 | 2 小时 | ¥3 |
| 完整模式 | 3 小时 | ¥4.5 |

---

## 📁 输出文件

实验完成后，结果保存在 `results_gpu/` 目录：

```
results_gpu/
├── all_results_20241221_123456.csv     # 所有结果表格
├── report_20241221_123456.json         # 详细 JSON 报告
└── ...
```

### 结果字段说明

| 字段 | 说明 |
|------|------|
| experiment | 实验编号 (exp1/exp2) |
| dataset | 数据集 (ml-100k/ml-1m) |
| model | 模型名称 |
| test_auc | 测试集 AUC |
| train_time_sec | 训练时间（秒） |
| qps | 推理速度（样本/秒） |

---

## 🔍 常见问题

### Q: CUDA out of memory
```bash
# 减小 batch size
python run_all_gpu.py --quick  # 使用 quick 模式
```

### Q: 数据下载失败
```bash
# 手动下载数据集
cd data
mkdir -p ml-100k ml-1m
# 从 https://grouplens.org/datasets/movielens/ 下载
```

### Q: 找不到模块
```bash
# 确保在正确目录
cd /path/to/standalone_din
python -c "from models import DINRichLite; print('OK')"
```

---

## 📞 联系方式

如有问题，请检查：
1. GPU 是否正确识别：`python -c "import torch; print(torch.cuda.is_available())"`
2. 依赖是否安装完整：`pip list | grep -E "torch|numpy|pandas|lightgbm"`
3. 文件结构是否正确：`ls -la` 确认文件存在
