#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合精排模块

创新点: DIN 深度特征 + LightGBM 精排

架构:
1. DIN 模型提取用户兴趣向量（注意力加权后的历史表示）
2. 将 DIN 提取的向量作为特征输入 LightGBM
3. LightGBM 结合深度特征 + 交叉特征进行最终排序

优势:
- 保留 DIN 的深度语义特征
- LightGBM 擅长处理交叉特征和特征工程
- 结合深度学习和树模型的优点
- 线上部署灵活（可以预计算 DIN embedding）
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("警告: LightGBM 未安装，混合精排功能不可用")


class DINEmbeddingExtractor(nn.Module):
    """
    DIN 嵌入提取器
    
    从训练好的 DIN 模型中提取用户兴趣向量。
    """
    
    def __init__(self, din_model, device='cpu'):
        super(DINEmbeddingExtractor, self).__init__()
        self.din_model = din_model
        self.device = device
        self.din_model.to(device)
        self.din_model.eval()
    
    @torch.no_grad()
    def extract_embeddings(self, data_loader, include_model_score=True):
        """
        从数据加载器中提取所有样本的 DIN 嵌入
        
        Returns:
            embeddings: [N, embedding_dim] 用户兴趣向量
            scores: [N] DIN 模型输出的分数
            labels: [N] 真实标签
            extra_features: [N, extra_dim] 额外特征（用户/物品特征）
        """
        all_embeddings = []
        all_scores = []
        all_labels = []
        all_extra_features = []
        
        for batch in tqdm(data_loader, desc="提取 DIN 嵌入"):
            # 移动到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 提取嵌入和额外特征
            emb, extra = self._extract_single_batch(batch)
            all_embeddings.append(emb.cpu().numpy())
            all_extra_features.append(extra)
            
            if include_model_score:
                score = self.din_model(batch)
                all_scores.append(torch.sigmoid(score).cpu().numpy())
            
            all_labels.append(batch['label'].cpu().numpy())
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        extra_features = np.concatenate(all_extra_features, axis=0)
        
        if include_model_score:
            scores = np.concatenate(all_scores, axis=0)
        else:
            scores = None
        
        return embeddings, scores, labels, extra_features
    
    def _extract_single_batch(self, batch):
        """
        从单个 batch 提取嵌入
        
        需要根据模型类型处理
        """
        # 获取序列嵌入
        item_seq = batch['item_seq']
        seq_emb = self.din_model.item_embedding(item_seq)  # [B, L, D]
        target_emb = self.din_model.item_embedding(batch['target_item'])  # [B, D]
        
        # 如果模型有注意力层，使用注意力加权
        if hasattr(self.din_model, 'attention'):
            user_interest, _ = self.din_model.attention(
                target_emb, seq_emb, batch['item_seq_mask']
            )
        else:
            # 简单平均
            mask = batch['item_seq_mask'].unsqueeze(-1)
            user_interest = (seq_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # 提取额外特征
        extra_features = self._extract_extra_features(batch)
        
        return user_interest, extra_features
    
    def _extract_extra_features(self, batch):
        """提取额外的特征用于 LightGBM"""
        batch_size = batch['user_id'].shape[0]
        
        # 收集所有特征
        features = []
        
        # 用户特征
        for key in ['user_id', 'user_age', 'user_gender', 'user_occupation', 'user_activity']:
            if key in batch:
                features.append(batch[key].cpu().numpy().reshape(-1, 1))
        
        # 物品特征
        for key in ['target_item', 'item_genre', 'item_year', 'item_popularity']:
            if key in batch:
                val = batch.get(key, batch.get('target_item'))
                features.append(val.cpu().numpy().reshape(-1, 1))
        
        # 时间特征
        for key in ['time_hour', 'time_dow', 'time_weekend']:
            if key in batch:
                features.append(batch[key].cpu().numpy().reshape(-1, 1))
        
        # 序列统计特征
        if 'seq_len' in batch:
            features.append(batch['seq_len'].cpu().numpy().reshape(-1, 1))
        
        return np.hstack(features) if features else np.zeros((batch_size, 1))


class HybridRanker:
    """
    混合精排器
    
    结合 DIN 深度特征和 LightGBM 进行精排。
    """
    
    def __init__(
        self,
        din_model,
        device='cpu',
        lgb_params=None,
        use_din_score=True
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
        
        self.din_extractor = DINEmbeddingExtractor(din_model, device)
        self.use_din_score = use_din_score
        self.device = device
        self.lgb_model = None
        
        # LightGBM 默认参数
        self.lgb_params = lgb_params or {
            'objective': 'binary',
            'metric': ['auc', 'binary_logloss'],
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 2020
        }
    
    def fit(
        self,
        train_loader,
        valid_loader,
        num_boost_round=500,
        early_stopping_rounds=50
    ):
        """
        训练混合精排模型
        """
        print("=" * 60)
        print("训练混合精排模型")
        print("=" * 60)
        
        # 1. 提取训练集的 DIN 嵌入
        print("\n1. 提取训练集 DIN 嵌入...")
        train_emb, train_din_scores, train_labels, train_extra = \
            self.din_extractor.extract_embeddings(train_loader, include_model_score=self.use_din_score)
        
        # 2. 提取验证集的 DIN 嵌入
        print("\n2. 提取验证集 DIN 嵌入...")
        valid_emb, valid_din_scores, valid_labels, valid_extra = \
            self.din_extractor.extract_embeddings(valid_loader, include_model_score=self.use_din_score)
        
        # 3. 构建 LightGBM 特征
        print("\n3. 构建 LightGBM 特征...")
        train_features = self._build_lgb_features(train_emb, train_din_scores, train_extra)
        valid_features = self._build_lgb_features(valid_emb, valid_din_scores, valid_extra)
        
        print(f"   训练集: {train_features.shape}")
        print(f"   验证集: {valid_features.shape}")
        
        # 4. 训练 LightGBM
        print("\n4. 训练 LightGBM...")
        train_data = lgb.Dataset(train_features, label=train_labels)
        valid_data = lgb.Dataset(valid_features, label=valid_labels, reference=train_data)
        
        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(100)
        ]
        
        self.lgb_model = lgb.train(
            self.lgb_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=callbacks
        )
        
        # 5. 评估
        print("\n5. 训练完成!")
        print(f"   最佳迭代: {self.lgb_model.best_iteration}")
        
        # 保存验证集结果供后续分析
        self.valid_predictions = self.lgb_model.predict(valid_features)
        self.valid_labels = valid_labels
        self.valid_din_scores = valid_din_scores
        
        return self
    
    def evaluate(self, data_loader):
        """
        评估混合精排模型
        """
        if self.lgb_model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        # 提取特征
        emb, din_scores, labels, extra = \
            self.din_extractor.extract_embeddings(data_loader, include_model_score=self.use_din_score)
        features = self._build_lgb_features(emb, din_scores, extra)
        
        # 预测
        predictions = self.lgb_model.predict(features)
        
        # 计算指标
        auc = roc_auc_score(labels, predictions)
        logloss = log_loss(labels, predictions)
        
        return {
            'auc': auc,
            'logloss': logloss,
            'predictions': predictions,
            'labels': labels,
            'din_scores': din_scores
        }
    
    def _build_lgb_features(self, embeddings, din_scores, extra_features):
        """
        构建 LightGBM 输入特征
        
        特征组成:
        - DIN embedding（降维或直接使用）
        - DIN 预测分数
        - 用户/物品/时间特征
        """
        features_list = [embeddings]
        
        if din_scores is not None:
            features_list.append(din_scores.reshape(-1, 1))
        
        features_list.append(extra_features)
        
        return np.hstack(features_list)
    
    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if self.lgb_model is None:
            raise RuntimeError("模型未训练")
        
        importance = self.lgb_model.feature_importance(importance_type='gain')
        
        # 构建特征名称
        n_emb = self.lgb_model.num_feature() - 1 - 10  # 假设有 10 个额外特征
        feature_names = [f'din_emb_{i}' for i in range(n_emb)]
        feature_names.append('din_score')
        feature_names.extend([
            'user_id', 'user_age', 'user_gender', 'user_occupation', 'user_activity',
            'item_id', 'item_genre', 'item_year', 'item_popularity',
            'time_hour', 'time_dow', 'time_weekend', 'seq_len'
        ][:len(importance) - n_emb - 1])
        
        # 排序
        indices = np.argsort(importance)[::-1][:top_n]
        
        return [(feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                 importance[i]) for i in indices]
    
    def compare_with_din(self):
        """
        比较混合模型与纯 DIN 的效果
        """
        if self.valid_predictions is None:
            raise RuntimeError("请先调用 fit()")
        
        # 混合模型
        hybrid_auc = roc_auc_score(self.valid_labels, self.valid_predictions)
        hybrid_logloss = log_loss(self.valid_labels, self.valid_predictions)
        
        # 纯 DIN
        din_auc = roc_auc_score(self.valid_labels, self.valid_din_scores)
        din_logloss = log_loss(self.valid_labels, self.valid_din_scores)
        
        print("\n" + "=" * 60)
        print("混合模型 vs 纯 DIN 对比")
        print("=" * 60)
        print(f"{'模型':<15} {'AUC':<12} {'LogLoss':<12}")
        print("-" * 40)
        print(f"{'DIN':<15} {din_auc:<12.4f} {din_logloss:<12.4f}")
        print(f"{'Hybrid':<15} {hybrid_auc:<12.4f} {hybrid_logloss:<12.4f}")
        print("-" * 40)
        
        auc_improvement = (hybrid_auc - din_auc) / din_auc * 100
        print(f"AUC 提升: {auc_improvement:+.2f}%")
        
        return {
            'din': {'auc': din_auc, 'logloss': din_logloss},
            'hybrid': {'auc': hybrid_auc, 'logloss': hybrid_logloss},
            'auc_improvement': auc_improvement
        }


def train_hybrid_ranker(
    din_model,
    train_loader,
    valid_loader,
    test_loader,
    device='cpu'
):
    """
    便捷函数：训练混合精排模型并评估
    """
    print("\n" + "=" * 80)
    print("混合精排: DIN + LightGBM")
    print("=" * 80)
    
    # 创建混合精排器
    ranker = HybridRanker(din_model, device=device)
    
    # 训练
    ranker.fit(train_loader, valid_loader)
    
    # 与纯 DIN 对比
    comparison = ranker.compare_with_din()
    
    # 测试集评估
    print("\n测试集评估:")
    test_results = ranker.evaluate(test_loader)
    print(f"  Hybrid AUC: {test_results['auc']:.4f}")
    print(f"  Hybrid LogLoss: {test_results['logloss']:.4f}")
    
    # 特征重要性
    print("\n特征重要性 Top 10:")
    for name, imp in ranker.get_feature_importance(10):
        print(f"  {name}: {imp:.2f}")
    
    return ranker, test_results


if __name__ == "__main__":
    print("混合精排模块测试")
    print("=" * 60)
    
    if not HAS_LIGHTGBM:
        print("请安装 LightGBM: pip install lightgbm")
    else:
        print("LightGBM 已安装，混合精排功能可用")
        print("\n使用示例:")
        print("  from hybrid_ranker import HybridRanker, train_hybrid_ranker")
        print("  ranker = HybridRanker(din_model, device='cuda')")
        print("  ranker.fit(train_loader, valid_loader)")
        print("  results = ranker.evaluate(test_loader)")
