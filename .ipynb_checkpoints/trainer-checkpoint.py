#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版训练器

支持丰富特征的模型训练。
支持 CTR 指标（AUC, LogLoss）和 Top-K 推荐指标（Recall@K, NDCG@K, HR@K, MRR）。
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from tqdm import tqdm
import time


# ========================================
# Top-K 评估指标
# ========================================

def hit_at_k(ranked_items, ground_truth, k):
    """
    Hit Rate @ K
    如果 ground_truth 在 top-k 中，返回 1，否则返回 0
    """
    return 1.0 if ground_truth in ranked_items[:k] else 0.0


def recall_at_k(ranked_items, ground_truth, k):
    """
    Recall @ K
    对于单个 ground truth，等同于 Hit Rate
    """
    return hit_at_k(ranked_items, ground_truth, k)


def ndcg_at_k(ranked_items, ground_truth, k):
    """
    NDCG @ K (Normalized Discounted Cumulative Gain)
    """
    for i, item in enumerate(ranked_items[:k]):
        if item == ground_truth:
            # DCG = 1 / log2(rank + 1)，IDCG = 1 / log2(2) = 1
            return 1.0 / np.log2(i + 2)  # +2 因为 rank 从 1 开始
    return 0.0


def mrr_at_k(ranked_items, ground_truth, k):
    """
    MRR @ K (Mean Reciprocal Rank)
    """
    for i, item in enumerate(ranked_items[:k]):
        if item == ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_items, ground_truth, k):
    """
    Precision @ K
    对于单个 ground truth: 命中则为 1/k，否则为 0
    """
    if ground_truth in ranked_items[:k]:
        return 1.0 / k
    return 0.0


class RichTrainer:
    """
    增强版训练器
    
    支持 batch 字典形式的输入。
    支持多 GPU DataParallel 加速。
    """
    
    def __init__(
        self,
        model,
        device='cpu',
        learning_rate=1e-3,
        weight_decay=1e-5,
        use_multi_gpu=False  # 是否使用多 GPU
    ):
        self.device = device
        self.use_multi_gpu = use_multi_gpu and torch.cuda.device_count() > 1
        
        # 将模型移到设备
        model = model.to(device)
        
        # 多 GPU 支持
        if self.use_multi_gpu:
            print(f"🔥 使用 DataParallel: {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
        
        self.model = model
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    @property
    def raw_model(self):
        """获取原始模型（用于访问模型属性或保存）"""
        if self.use_multi_gpu and hasattr(self.model, 'module'):
            return self.model.module
        return self.model
    
    def _move_batch_to_device(self, batch):
        """将 batch 移动到设备"""
        return {k: v.to(self.device) for k, v in batch.items()}
    
    def train_epoch(self, train_loader, show_progress=True):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        iterator = tqdm(train_loader, desc='Training') if show_progress else train_loader
        
        for batch in iterator:
            batch = self._move_batch_to_device(batch)
            
            self.optimizer.zero_grad()
            
            logits = self.model(batch)
            loss = self.criterion(logits, batch['label'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, data_loader, show_progress=False):
        """评估模型"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        iterator = tqdm(data_loader, desc='Evaluating') if show_progress else data_loader
        
        with torch.no_grad():
            for batch in iterator:
                batch = self._move_batch_to_device(batch)
                
                logits = self.model(batch)
                preds = torch.sigmoid(logits).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        auc = roc_auc_score(all_labels, all_preds)
        logloss = log_loss(all_labels, np.clip(all_preds, 1e-7, 1-1e-7))
        
        return {
            'auc': auc,
            'logloss': logloss
        }
    
    def fit(
        self,
        train_loader,
        valid_loader,
        epochs=20,
        early_stopping_patience=5,
        show_progress=True
    ):
        """训练模型"""
        best_valid_auc = 0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, show_progress)
            valid_metrics = self.evaluate(valid_loader)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - "
                  f"Valid AUC: {valid_metrics['auc']:.4f} - "
                  f"Valid LogLoss: {valid_metrics['logloss']:.4f}")
            
            if valid_metrics['auc'] > best_valid_auc:
                best_valid_auc = valid_metrics['auc']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_valid_auc': best_valid_auc,
            'final_epoch': epoch + 1
        }
    
    def evaluate_topk(
        self,
        eval_data,
        feature_processor,
        interaction_extractor,
        max_seq_length,
        ks=[5, 10, 20],
        show_progress=True,
        batch_size=256
    ):
        """
        Top-K 推荐评估（批量优化版）
        
        Args:
            eval_data: list of dict，来自 get_topk_eval_data
            feature_processor: 特征处理器
            interaction_extractor: 交互特征提取器
            max_seq_length: 最大序列长度
            ks: 评估的 K 值列表
            show_progress: 是否显示进度条
            batch_size: 批量评估的用户数
        
        Returns:
            dict: 各指标在不同 K 下的值
        """
        from data_loader import build_topk_batch_multi
        
        self.model.eval()
        
        # 初始化指标累加器
        all_hr = {k: [] for k in ks}
        all_ndcg = {k: [] for k in ks}
        all_mrr = {k: [] for k in ks}
        
        # 分批处理
        num_users = len(eval_data)
        num_batches = (num_users + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc='Top-K Eval')
        
        with torch.no_grad():
            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_users)
                batch_eval_data = eval_data[start_idx:end_idx]
                
                # 批量构建并评估
                for eval_item in batch_eval_data:
                    # 构建单用户的候选 batch
                    batch = build_topk_batch_multi(
                        eval_item, feature_processor, interaction_extractor,
                        max_seq_length, self.device
                    )
                    
                    # 预测分数
                    logits = self.model(batch)
                    scores = torch.sigmoid(logits).cpu().numpy()
                    
                    # 排序
                    candidates = eval_item['candidates']
                    ground_truth = eval_item['ground_truth']
                    sorted_indices = np.argsort(-scores)
                    ranked_items = [candidates[i] for i in sorted_indices]
                    
                    # 计算指标
                    for k in ks:
                        all_hr[k].append(hit_at_k(ranked_items, ground_truth, k))
                        all_ndcg[k].append(ndcg_at_k(ranked_items, ground_truth, k))
                        all_mrr[k].append(mrr_at_k(ranked_items, ground_truth, k))
        
        # 计算平均值
        results = {}
        for k in ks:
            results[f'HR@{k}'] = np.mean(all_hr[k])
            results[f'Recall@{k}'] = np.mean(all_hr[k])  # 单 GT 等于 HR
            results[f'NDCG@{k}'] = np.mean(all_ndcg[k])
            results[f'MRR@{k}'] = np.mean(all_mrr[k])
            results[f'Precision@{k}'] = np.mean(all_hr[k]) / k
        
        return results


def measure_inference_speed_rich(model, data_loader, device='cpu', warmup_batches=5, measure_batches=20):
    """
    测量推理速度（QPS）
    
    适用于 batch 字典输入的模型。
    """
    model.eval()
    model = model.to(device)
    
    sample_batch = next(iter(data_loader))
    batch_size = sample_batch['user_id'].shape[0]
    
    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= warmup_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch)
    
    # 测量
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= measure_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            _ = model(batch)
            total_samples += batch['user_id'].shape[0]
    
    elapsed = time.time() - start_time
    qps = total_samples / elapsed if elapsed > 0 else 0
    
    return {
        'qps': qps,
        'total_samples': total_samples,
        'elapsed_time': elapsed
    }


if __name__ == "__main__":
    print("测试增强版训练器...")
    
    from data_loader import get_rich_dataloaders
    from models import DINRichLite
    
    train_loader, valid_loader, test_loader, info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name='ml-100k',
        max_seq_length=50,
        batch_size=256
    )
    
    model = DINRichLite(
        num_items=info['num_items'],
        num_users=info['num_users'],
        feature_dims=info['feature_dims'],
        embedding_dim=64
    )
    
    trainer = RichTrainer(model=model, device='cpu')
    
    # 快速测试
    result = trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=2,
        show_progress=True
    )
    
    print(f"\n训练结果: {result}")
    
    test_metrics = trainer.evaluate(test_loader)
    print(f"测试结果: {test_metrics}")
