#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云端 GPU 完整实验脚本

适合在 AutoDL / Colab / 阿里云等 GPU 环境运行。
支持 ml-100k 和 ml-1m 双数据集。
包含全部三个实验。

使用方法:
    python run_all_gpu.py                    # 运行所有实验（两个数据集）
    python run_all_gpu.py --dataset ml-100k  # 只运行 ml-100k
    python run_all_gpu.py --dataset ml-1m    # 只运行 ml-1m
    python run_all_gpu.py --quick            # 快速测试模式
    
    # 单独运行某个实验
    python run_all_gpu.py --exp 1            # 只运行实验1
    python run_all_gpu.py --exp 2            # 只运行实验2
    python run_all_gpu.py --exp 3            # 只运行实验3
    python run_all_gpu.py --exp 1,2          # 运行实验1和2

预估时间 (GPU, 两个数据集):
    实验1（序列长度+模型对比）: 约 40-60 分钟
    实验2（方法对比+混合精排）: 约 30-40 分钟
    实验3（消融实验）:          约 20-30 分钟
    总计:                       约 1.5-2.5 小时
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from tqdm import tqdm

from data_loader import get_rich_dataloaders
from models import DINRichLite, SimpleAveragePoolingRich, GRU4Rec, SASRec, NARM, AttentionLayer
from trainer import RichTrainer, measure_inference_speed_rich
from feature_engineering import FeatureProcessor, InteractionFeatureExtractor, prepare_lightgbm_features

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("⚠️ LightGBM 未安装，混合精排将跳过")

# ========================================
# 配置
# ========================================

parser = argparse.ArgumentParser(description='云端 GPU 完整实验')
parser.add_argument('--dataset', type=str, default='both', 
                    choices=['ml-100k', 'ml-1m', 'both'],
                    help='数据集选择')
parser.add_argument('--quick', action='store_true', 
                    help='快速测试模式（减少 epochs 和序列长度）')
parser.add_argument('--epochs', type=int, default=50,
                    help='训练轮数（默认 50）')
parser.add_argument('--exp', type=str, default='all',
                    help='运行哪些实验: 1, 2, 3, 1,2, 1,3, 2,3, all')
args = parser.parse_args()

# 解析要运行的实验
if args.exp == 'all':
    EXPERIMENTS_TO_RUN = [1, 2, 3]
else:
    EXPERIMENTS_TO_RUN = [int(x.strip()) for x in args.exp.split(',')]

# 设备检测
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实验参数
if args.quick:
    EPOCHS = 10
    SEQ_LENGTHS = [20, 50]
    BATCH_SIZE = 512
else:
    EPOCHS = args.epochs
    SEQ_LENGTHS = [20, 50, 100, 150]
    BATCH_SIZE = 512 if DEVICE == 'cuda' else 256

EMBEDDING_DIM = 64
MODELS_TO_TEST = ['DIN', 'GRU4Rec', 'SASRec', 'NARM', 'AvgPool']

# 数据集
if args.dataset == 'both':
    DATASETS = ['ml-100k', 'ml-1m']
else:
    DATASETS = [args.dataset]

# 结果目录
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results_gpu')
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 80)
print("🚀 云端 GPU 完整实验")
print("=" * 80)
print(f"设备: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"数据集: {DATASETS}")
print(f"实验: {EXPERIMENTS_TO_RUN}")
print(f"Epochs: {EPOCHS}")
print(f"序列长度: {SEQ_LENGTHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"模型: {MODELS_TO_TEST}")
print(f"快速模式: {args.quick}")
print("=" * 80)


# ========================================
# 消融实验的注意力变体
# ========================================

class TimeDecayRichAttention(nn.Module):
    """时间衰减注意力"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], time_decay=0.1):
        super().__init__()
        self.time_decay = time_decay
        mlp_input = 4 * input_dim
        layers = []
        prev_dim = mlp_input
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.attention_mlp = nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_mask=None):
        batch_size, seq_len, dim = keys.shape
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        attention_scores = self.attention_mlp(attention_input).squeeze(-1)
        
        positions = torch.arange(seq_len, device=keys.device).float()
        time_weights = torch.exp(self.time_decay * (positions - seq_len + 1))
        attention_scores = attention_scores * time_weights.unsqueeze(0)
        
        if keys_mask is not None:
            attention_scores = attention_scores.masked_fill(~keys_mask.bool(), -1e9)
        attention_weights = F.softmax(attention_scores, dim=-1)
        weighted_sum = torch.sum(attention_weights.unsqueeze(-1) * keys, dim=1)
        return weighted_sum, attention_weights


class MultiHeadRichAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, input_dim, num_heads=4, hidden_dims=[64, 32]):
        super().__init__()
        self.num_heads = num_heads
        self.attention_heads = nn.ModuleList([
            self._build_attention_mlp(4 * input_dim, hidden_dims)
            for _ in range(num_heads)
        ])
        self.output_proj = nn.Linear(input_dim, input_dim)
    
    def _build_attention_mlp(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.PReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, query, keys, keys_mask=None):
        batch_size, seq_len, dim = keys.shape
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        attention_input = torch.cat([
            keys, query_expanded,
            keys * query_expanded,
            keys - query_expanded
        ], dim=-1)
        
        head_outputs = []
        for head in self.attention_heads:
            scores = head(attention_input).squeeze(-1)
            if keys_mask is not None:
                scores = scores.masked_fill(~keys_mask.bool(), -1e9)
            weights = F.softmax(scores, dim=-1)
            output = torch.sum(weights.unsqueeze(-1) * keys, dim=1)
            head_outputs.append(output)
        
        combined = torch.stack(head_outputs, dim=1).mean(dim=1)
        return self.output_proj(combined), None


class DINRichVariant(nn.Module):
    """DIN 消融变体"""
    
    def __init__(self, num_items, num_users, feature_dims, embedding_dim=64,
                 attention_type='base', enhanced_mlp=False):
        super().__init__()
        self.attention_type = attention_type
        self.enhanced_mlp = enhanced_mlp
        
        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim, padding_idx=0)
        
        self.feature_embeddings = nn.ModuleDict()
        for name, num_values in feature_dims.items():
            self.feature_embeddings[name] = nn.Embedding(num_values + 1, embedding_dim // 4, padding_idx=0)
        
        feature_embed_dim = (embedding_dim // 4) * len(feature_dims)
        self.total_embed_dim = embedding_dim + feature_embed_dim
        
        # 选择注意力类型
        if attention_type == 'time_decay':
            self.attention = TimeDecayRichAttention(self.total_embed_dim)
        elif attention_type == 'multi_head':
            self.attention = MultiHeadRichAttention(self.total_embed_dim, num_heads=4)
        else:
            self.attention = AttentionLayer(self.total_embed_dim)
        
        # MLP
        mlp_input_dim = self.total_embed_dim * 3 + embedding_dim
        if enhanced_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 256),
                nn.BatchNorm1d(256),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.PReLU(),
                nn.Linear(64, 1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, 128),
                nn.PReLU(),
                nn.Linear(128, 64),
                nn.PReLU(),
                nn.Linear(64, 1)
            )
    
    def _get_rich_embedding(self, item_ids, features):
        item_emb = self.item_embedding(item_ids)
        feature_embs = []
        for name, emb_layer in self.feature_embeddings.items():
            if name in features:
                feature_embs.append(emb_layer(features[name]))
        if feature_embs:
            if len(item_emb.shape) == 2:
                feature_cat = torch.cat(feature_embs, dim=-1)
            else:
                feature_cat = torch.cat(feature_embs, dim=-1)
            return torch.cat([item_emb, feature_cat], dim=-1)
        return item_emb
    
    def forward(self, batch):
        item_seq = batch['item_seq']
        target_item = batch['target_item']
        user_id = batch['user_id']
        seq_mask = (item_seq > 0).float()
        
        seq_features = {k: v for k, v in batch.items() 
                       if k.endswith('_seq') and k != 'item_seq'}
        seq_emb = self._get_rich_embedding(item_seq, seq_features)
        
        target_features = {k.replace('target_', ''): v for k, v in batch.items() 
                          if k.startswith('target_') and k != 'target_item'}
        target_emb = self._get_rich_embedding(target_item, target_features)
        
        user_emb = self.user_embedding(user_id)
        
        interest_emb, _ = self.attention(target_emb, seq_emb, seq_mask)
        
        seq_mean = (seq_emb * seq_mask.unsqueeze(-1)).sum(dim=1) / (seq_mask.sum(dim=1, keepdim=True) + 1e-8)
        
        mlp_input = torch.cat([interest_emb, target_emb, seq_mean, user_emb], dim=-1)
        logits = self.mlp(mlp_input).squeeze(-1)
        return logits


# ========================================
# 混合精排模块
# ========================================

class HybridRanker:
    """DIN + LightGBM 混合精排"""
    
    def __init__(self, din_model, device='cpu'):
        self.din_model = din_model
        self.device = device
        self.lgb_model = None
    
    @torch.no_grad()
    def extract_din_features(self, data_loader):
        """提取 DIN 嵌入作为特征"""
        self.din_model.eval()
        self.din_model.to(self.device)
        
        all_embeddings = []
        all_scores = []
        all_labels = []
        
        for batch in data_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 获取嵌入
            item_seq = batch['item_seq']
            seq_emb = self.din_model.item_embedding(item_seq)
            target_emb = self.din_model.item_embedding(batch['target_item'])
            user_emb = self.din_model.user_embedding(batch['user_id'])
            
            seq_mask = (item_seq > 0).float()
            seq_mean = (seq_emb * seq_mask.unsqueeze(-1)).sum(dim=1) / (seq_mask.sum(dim=1, keepdim=True) + 1e-8)
            
            # 拼接特征
            features = torch.cat([target_emb, user_emb, seq_mean], dim=-1)
            all_embeddings.append(features.cpu().numpy())
            
            # DIN 分数
            score = torch.sigmoid(self.din_model(batch))
            all_scores.append(score.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        # 拼接 DIN 分数作为特征
        features = np.column_stack([embeddings, scores])
        return features, labels
    
    def train_lgb(self, train_loader, valid_loader):
        """训练 LightGBM"""
        if not HAS_LIGHTGBM:
            return None
        
        X_train, y_train = self.extract_din_features(train_loader)
        X_valid, y_valid = self.extract_din_features(valid_loader)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'verbose': -1,
            'random_state': 2020
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid)
        
        self.lgb_model = lgb.train(
            params, train_data,
            num_boost_round=300,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        
        return self.lgb_model
    
    def evaluate(self, test_loader):
        """评估混合模型"""
        from sklearn.metrics import roc_auc_score, log_loss
        
        X_test, y_test = self.extract_din_features(test_loader)
        y_pred = self.lgb_model.predict(X_test)
        
        auc = roc_auc_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        
        return {'auc': auc, 'logloss': logloss}


# ========================================
# 实验一：序列长度敏感性 + 模型对比
# ========================================

def run_experiment1(dataset_name):
    """实验一：不同序列长度下各模型的表现"""
    print("\n" + "=" * 80)
    print(f"📊 实验一：序列长度敏感性 + 模型对比 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    
    for seq_length in SEQ_LENGTHS:
        print(f"\n🔬 序列长度: {seq_length}")
        
        train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            batch_size=BATCH_SIZE
        )
        
        for model_name in MODELS_TO_TEST:
            print(f"  🚀 {model_name}...", end=" ", flush=True)
            
            try:
                if model_name == 'DIN':
                    model = DINRichLite(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM
                    )
                elif model_name == 'GRU4Rec':
                    model = GRU4Rec(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM
                    )
                elif model_name == 'SASRec':
                    model = SASRec(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM,
                        num_heads=2,
                        num_layers=2,
                        max_seq_len=seq_length
                    )
                elif model_name == 'NARM':
                    model = NARM(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM,
                        hidden_dim=EMBEDDING_DIM
                    )
                elif model_name == 'AvgPool':
                    model = SimpleAveragePoolingRich(
                        num_items=dataset_info['num_items'],
                        num_users=dataset_info['num_users'],
                        feature_dims=dataset_info['feature_dims'],
                        embedding_dim=EMBEDDING_DIM
                    )
                
                trainer = RichTrainer(model=model, device=DEVICE)
                t1 = time.time()
                train_result = trainer.fit(
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    epochs=EPOCHS,
                    early_stopping_patience=10,
                    show_progress=False
                )
                train_time = time.time() - t1
                
                test_metrics = trainer.evaluate(test_loader)
                speed = measure_inference_speed_rich(model, test_loader, DEVICE)
                
                results.append({
                    'experiment': 'exp1_seq_model',
                    'dataset': dataset_name,
                    'seq_length': seq_length,
                    'model': model_name,
                    'test_auc': test_metrics['auc'],
                    'test_logloss': test_metrics['logloss'],
                    'best_valid_auc': train_result['best_valid_auc'],
                    'train_time_sec': train_time,
                    'qps': speed['qps'],
                    'num_params': sum(p.numel() for p in model.parameters()),
                    'status': 'success'
                })
                
                print(f"AUC={test_metrics['auc']:.4f}, Time={train_time:.1f}s")
                
            except Exception as e:
                print(f"❌ {str(e)[:50]}")
                results.append({
                    'experiment': 'exp1_seq_model',
                    'dataset': dataset_name,
                    'seq_length': seq_length,
                    'model': model_name,
                    'test_auc': None,
                    'status': f'error: {str(e)[:100]}'
                })
    
    return results


# ========================================
# 实验二：方法对比 + LightGBM + 混合精排
# ========================================

def run_experiment2(dataset_name):
    """实验二：DIN vs 传统方法 + 混合精排"""
    print("\n" + "=" * 80)
    print(f"📊 实验二：方法对比 + 混合精排 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    seq_length = 50
    
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        batch_size=BATCH_SIZE
    )
    
    din_model = None  # 保存用于混合精排
    
    # 测试各深度模型
    for model_name in MODELS_TO_TEST:
        print(f"  🚀 {model_name}...", end=" ", flush=True)
        
        try:
            if model_name == 'DIN':
                model = DINRichLite(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM
                )
                din_model = model  # 保存
            elif model_name == 'GRU4Rec':
                model = GRU4Rec(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM,
                    hidden_dim=EMBEDDING_DIM
                )
            elif model_name == 'SASRec':
                model = SASRec(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM,
                    num_heads=2,
                    num_layers=2,
                    max_seq_len=seq_length
                )
            elif model_name == 'NARM':
                model = NARM(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM,
                    hidden_dim=EMBEDDING_DIM
                )
            elif model_name == 'AvgPool':
                model = SimpleAveragePoolingRich(
                    num_items=dataset_info['num_items'],
                    num_users=dataset_info['num_users'],
                    feature_dims=dataset_info['feature_dims'],
                    embedding_dim=EMBEDDING_DIM
                )
            
            trainer = RichTrainer(model=model, device=DEVICE)
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=EPOCHS,
                early_stopping_patience=10,
                show_progress=False
            )
            train_time = time.time() - t1
            
            test_metrics = trainer.evaluate(test_loader)
            speed = measure_inference_speed_rich(model, test_loader, DEVICE)
            
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': model_name,
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'train_time_sec': train_time,
                'qps': speed['qps'],
                'status': 'success'
            })
            
            print(f"AUC={test_metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': model_name,
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # LightGBM 单独
    if HAS_LIGHTGBM:
        print("  🚀 LightGBM (pure)...", end=" ", flush=True)
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            from sklearn.model_selection import train_test_split
            
            data_path = os.path.join('./data', dataset_name)
            if dataset_name == 'ml-100k':
                interactions = pd.read_csv(
                    os.path.join(data_path, 'u.data'),
                    sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp']
                )
            else:
                interactions = pd.read_csv(
                    os.path.join(data_path, 'ratings.dat'),
                    sep='::', names=['user_id', 'item_id', 'rating', 'timestamp'],
                    engine='python'
                )
            
            interaction_extractor = InteractionFeatureExtractor(interactions)
            X, y, feature_names = prepare_lightgbm_features(
                interactions, fp, interaction_extractor, max_seq_length=seq_length
            )
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)
            X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.125, random_state=2020)
            
            params = {
                'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
                'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                'verbose': -1, 'random_state': 2020
            }
            
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_valid, label=y_valid)
            
            t1 = time.time()
            lgb_model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            train_time = time.time() - t1
            
            y_pred = lgb_model.predict(X_test)
            test_auc = roc_auc_score(y_test, y_pred)
            
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': 'LightGBM',
                'test_auc': test_auc,
                'train_time_sec': train_time,
                'status': 'success'
            })
            print(f"AUC={test_auc:.4f}")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp2_method_compare',
                'dataset': dataset_name,
                'model': 'LightGBM',
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # 混合精排：DIN + LightGBM
    if HAS_LIGHTGBM and din_model is not None:
        print("  🚀 Hybrid (DIN + LightGBM)...", end=" ", flush=True)
        try:
            hybrid = HybridRanker(din_model, device=DEVICE)
            t1 = time.time()
            hybrid.train_lgb(train_loader, valid_loader)
            train_time = time.time() - t1
            
            test_metrics = hybrid.evaluate(test_loader)
            
            results.append({
                'experiment': 'exp2_hybrid',
                'dataset': dataset_name,
                'model': 'DIN+LightGBM',
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'train_time_sec': train_time,
                'status': 'success'
            })
            print(f"AUC={test_metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp2_hybrid',
                'dataset': dataset_name,
                'model': 'DIN+LightGBM',
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    return results


# ========================================
# 实验三：DIN 消融实验
# ========================================

def run_experiment3(dataset_name):
    """实验三：DIN 改进消融实验"""
    print("\n" + "=" * 80)
    print(f"📊 实验三：DIN 消融实验 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    seq_length = 50
    
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        batch_size=BATCH_SIZE
    )
    
    # 消融变体
    ablation_variants = [
        ('DIN-Base', 'base', False),
        ('DIN-TimeDec', 'time_decay', False),
        ('DIN-MultiHead', 'multi_head', False),
        ('DIN-Enhanced', 'base', True),
        ('DIN-Full', 'time_decay', True),
    ]
    
    for variant_name, attention_type, enhanced_mlp in ablation_variants:
        print(f"  🚀 {variant_name}...", end=" ", flush=True)
        
        try:
            model = DINRichVariant(
                num_items=dataset_info['num_items'],
                num_users=dataset_info['num_users'],
                feature_dims=dataset_info['feature_dims'],
                embedding_dim=EMBEDDING_DIM,
                attention_type=attention_type,
                enhanced_mlp=enhanced_mlp
            )
            
            trainer = RichTrainer(model=model, device=DEVICE)
            t1 = time.time()
            train_result = trainer.fit(
                train_loader=train_loader,
                valid_loader=valid_loader,
                epochs=EPOCHS,
                early_stopping_patience=10,
                show_progress=False
            )
            train_time = time.time() - t1
            
            test_metrics = trainer.evaluate(test_loader)
            speed = measure_inference_speed_rich(model, test_loader, DEVICE)
            
            results.append({
                'experiment': 'exp3_ablation',
                'dataset': dataset_name,
                'variant': variant_name,
                'attention_type': attention_type,
                'enhanced_mlp': enhanced_mlp,
                'test_auc': test_metrics['auc'],
                'test_logloss': test_metrics['logloss'],
                'best_valid_auc': train_result['best_valid_auc'],
                'train_time_sec': train_time,
                'qps': speed['qps'],
                'num_params': sum(p.numel() for p in model.parameters()),
                'status': 'success'
            })
            
            print(f"AUC={test_metrics['auc']:.4f}")
            
        except Exception as e:
            print(f"❌ {str(e)[:50]}")
            results.append({
                'experiment': 'exp3_ablation',
                'dataset': dataset_name,
                'variant': variant_name,
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    return results


# ========================================
# 主程序
# ========================================

if __name__ == '__main__':
    all_results = []
    experiment_start = datetime.now()
    
    print(f"\n⏰ 实验开始时间: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"📁 数据集: {dataset.upper()}")
        print(f"{'='*80}")
        
        if 1 in EXPERIMENTS_TO_RUN:
            results1 = run_experiment1(dataset)
            all_results.extend(results1)
        
        if 2 in EXPERIMENTS_TO_RUN:
            results2 = run_experiment2(dataset)
            all_results.extend(results2)
        
        if 3 in EXPERIMENTS_TO_RUN:
            results3 = run_experiment3(dataset)
            all_results.extend(results3)
    
    # 保存结果
    experiment_end = datetime.now()
    total_time = (experiment_end - experiment_start).total_seconds()
    
    df_results = pd.DataFrame(all_results)
    timestamp = experiment_start.strftime('%Y%m%d_%H%M%S')
    
    # CSV
    csv_file = os.path.join(RESULTS_DIR, f'all_results_{timestamp}.csv')
    df_results.to_csv(csv_file, index=False)
    
    # JSON 报告
    report = {
        'timestamp': timestamp,
        'device': DEVICE,
        'gpu_name': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU',
        'datasets': DATASETS,
        'experiments': EXPERIMENTS_TO_RUN,
        'epochs': EPOCHS,
        'seq_lengths': SEQ_LENGTHS,
        'models': MODELS_TO_TEST,
        'total_time_minutes': total_time / 60,
        'num_results': len(all_results),
        'results': all_results
    }
    
    json_file = os.path.join(RESULTS_DIR, f'report_{timestamp}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("📋 实验完成！")
    print("=" * 80)
    print(f"总耗时: {total_time/60:.1f} 分钟")
    print(f"实验数量: {len(all_results)}")
    print(f"\n📂 结果文件:")
    print(f"   {csv_file}")
    print(f"   {json_file}")
    
    # 各数据集最佳结果
    print("\n📊 各实验最佳 AUC:")
    df_success = df_results[df_results['status'] == 'success']
    
    for exp_name in df_success['experiment'].unique():
        df_exp = df_success[df_success['experiment'] == exp_name]
        if len(df_exp) > 0 and 'test_auc' in df_exp.columns:
            best = df_exp.loc[df_exp['test_auc'].idxmax()]
            model_col = 'model' if 'model' in best else 'variant'
            print(f"  {exp_name}: {best.get(model_col, 'N/A')} = {best['test_auc']:.4f}")
    
    print("=" * 80)
    print("✅ 所有实验完成！")
