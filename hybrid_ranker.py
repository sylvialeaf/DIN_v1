#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混合精排模块 (v2 - 修复版)

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

修复内容 (v2):
- 特征融合：DIN嵌入与LightGBM显式特征对齐
- Pipeline一致性：确保样本划分和特征schema一致
- 完整评估：添加Top-K评估指标 (HR@K, NDCG@K, MRR)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm
from collections import defaultdict

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("警告: LightGBM 未安装，混合精排功能不可用")


# ========================================
# 特征Schema定义 (确保一致性)
# ========================================

# 显式特征名称定义（与prepare_lightgbm_features对齐）
EXPLICIT_FEATURE_NAMES = [
    'user_age', 'user_gender', 'user_occupation', 'user_activity',
    'item_year', 'item_genre', 'item_genre_count', 'item_popularity',
    'seq_length', 'seq_unique_items', 'seq_avg_popularity', 'seq_recency',
    'time_hour', 'time_dow', 'time_weekend', 'time_position',
    'genre_match', 'year_match',
    'hist_genre_mean', 'hist_genre_std'
]


class DINEmbeddingExtractor(nn.Module):
    """
    DIN 嵌入提取器
    
    从训练好的 DIN 模型中提取用户兴趣向量。
    
    v2更新：支持特征处理器和交互提取器，确保特征schema一致
    """
    
    def __init__(self, din_model, device='cpu', feature_processor=None, interaction_extractor=None):
        super(DINEmbeddingExtractor, self).__init__()
        self.din_model = din_model
        self.device = device
        self.din_model.to(device)
        self.din_model.eval()
        
        # 特征处理器（用于提取显式特征）
        self.feature_processor = feature_processor
        self.interaction_extractor = interaction_extractor
    
    def set_feature_extractors(self, feature_processor, interaction_extractor):
        """设置特征处理器（延迟初始化）"""
        self.feature_processor = feature_processor
        self.interaction_extractor = interaction_extractor
    
    @torch.no_grad()
    def extract_embeddings(self, data_loader, include_model_score=True):
        """
        从数据加载器中提取所有样本的 DIN 嵌入
        
        Returns:
            embeddings: [N, embedding_dim] 用户兴趣向量
            scores: [N] DIN 模型输出的分数
            labels: [N] 真实标签
            extra_features: [N, extra_dim] 额外特征（与LightGBM特征schema对齐）
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
        
        # 提取额外特征（与LightGBM schema对齐）
        extra_features = self._extract_aligned_features(batch)
        
        return user_interest, extra_features
    
    def _extract_aligned_features(self, batch):
        """
        提取与LightGBM特征schema对齐的显式特征
        
        特征顺序与 prepare_lightgbm_features 完全一致！
        
        Returns:
            numpy array of shape [batch_size, 20]
        """
        batch_size = batch['user_id'].shape[0]
        
        features = []
        
        for i in range(batch_size):
            sample_features = self._extract_single_sample_features(batch, i)
            features.append(sample_features)
        
        return np.array(features)
    
    def _extract_single_sample_features(self, batch, idx):
        """提取单个样本的显式特征（20维）"""
        
        # 从batch中获取原始数据
        user_id = batch['user_id'][idx].cpu().item()
        target_item = batch['target_item'][idx].cpu().item()
        item_seq = batch['item_seq'][idx].cpu().numpy()
        item_seq_mask = batch['item_seq_mask'][idx].cpu().numpy()
        
        # 获取有效的历史序列
        valid_len = int(item_seq_mask.sum())
        history = item_seq[:valid_len].tolist() if valid_len > 0 else []
        
        # 使用特征处理器获取用户/物品特征
        if self.feature_processor is not None:
            user_feat = self.feature_processor.get_user_features(user_id)
            item_feat = self.feature_processor.get_item_features(target_item)
            
            # 历史物品特征
            hist_item_feats = [self.feature_processor.get_item_features(iid) for iid in history if iid > 0]
            hist_genres = [f['primary_genre'] for f in hist_item_feats]
            hist_years = [f['year_bucket'] for f in hist_item_feats]
        else:
            # 回退方案：直接从batch获取
            user_feat = {
                'age_bucket': batch.get('user_age', torch.zeros(1))[idx].cpu().item() if 'user_age' in batch else 0,
                'gender': batch.get('user_gender', torch.zeros(1))[idx].cpu().item() if 'user_gender' in batch else 0,
                'occupation': batch.get('user_occupation', torch.zeros(1))[idx].cpu().item() if 'user_occupation' in batch else 0
            }
            item_feat = {
                'year_bucket': batch.get('item_year', torch.zeros(1))[idx].cpu().item() if 'item_year' in batch else 0,
                'primary_genre': batch.get('item_genre', torch.zeros(1))[idx].cpu().item() if 'item_genre' in batch else 0,
                'genre_count': 1
            }
            hist_genres = []
            hist_years = []
        
        # 获取用户活跃度和物品热度
        if self.interaction_extractor is not None:
            user_activity = self.interaction_extractor.get_user_activity(user_id)
            item_popularity = self.interaction_extractor.get_item_popularity(target_item)
            
            # 序列特征
            seq_feat = {
                'seq_length': valid_len,
                'unique_items': len(set(history)) if history else 0,
                'avg_popularity': np.mean([self.interaction_extractor.get_item_popularity(iid) for iid in history]) if history else 0,
                'recency_weight': 0.5  # 简化处理
            }
            
            # 时间特征（如果batch中有timestamp）
            if 'timestamp' in batch:
                ts = batch['timestamp'][idx].cpu().item()
                time_feat = self.interaction_extractor.get_time_features(ts)
            else:
                time_feat = {'hour_bucket': 0, 'day_of_week': 0, 'is_weekend': 0, 'time_position': 0.5}
        else:
            user_activity = batch.get('user_activity', torch.zeros(1))[idx].cpu().item() if 'user_activity' in batch else 0
            item_popularity = batch.get('item_popularity', torch.zeros(1))[idx].cpu().item() if 'item_popularity' in batch else 0
            seq_feat = {'seq_length': valid_len, 'unique_items': len(set(history)) if history else 0, 'avg_popularity': 0, 'recency_weight': 0.5}
            time_feat = {
                'hour_bucket': batch.get('time_hour', torch.zeros(1))[idx].cpu().item() if 'time_hour' in batch else 0,
                'day_of_week': batch.get('time_dow', torch.zeros(1))[idx].cpu().item() if 'time_dow' in batch else 0,
                'is_weekend': batch.get('time_weekend', torch.zeros(1))[idx].cpu().item() if 'time_weekend' in batch else 0,
                'time_position': 0.5
            }
        
        # 构建特征向量（与prepare_lightgbm_features完全对齐，20维）
        sample = [
            # 用户特征 (4维)
            user_feat.get('age_bucket', 0),
            user_feat.get('gender', 0),
            user_feat.get('occupation', 0),
            user_activity,
            
            # 物品特征 (4维)
            item_feat.get('year_bucket', 0),
            item_feat.get('primary_genre', 0),
            item_feat.get('genre_count', 1),
            item_popularity,
            
            # 序列特征 (4维)
            seq_feat['seq_length'],
            seq_feat['unique_items'],
            seq_feat['avg_popularity'],
            seq_feat['recency_weight'],
            
            # 时间特征 (4维)
            time_feat['hour_bucket'],
            time_feat['day_of_week'],
            time_feat['is_weekend'],
            time_feat['time_position'],
            
            # 交叉特征 (2维)
            1 if item_feat.get('primary_genre', 0) in hist_genres else 0,  # 类型匹配
            1 if item_feat.get('year_bucket', 0) in hist_years else 0,     # 年代匹配
            
            # 历史统计 (2维)
            np.mean(hist_genres) if hist_genres else 0,
            np.std(hist_genres) if len(hist_genres) > 1 else 0,
        ]
        
        return sample


class HybridRanker:
    """
    混合精排器 (v2)
    
    结合 DIN 深度特征和 LightGBM 进行精排。
    
    v2更新：
    - 支持特征处理器，确保特征schema一致
    - 添加Top-K评估方法
    - 改进特征名称管理
    """
    
    def __init__(
        self,
        din_model,
        device='cpu',
        lgb_params=None,
        use_din_score=True,
        feature_processor=None,
        interaction_extractor=None
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM 未安装，请运行: pip install lightgbm")
        
        self.din_extractor = DINEmbeddingExtractor(
            din_model, device, feature_processor, interaction_extractor
        )
        self.use_din_score = use_din_score
        self.device = device
        self.lgb_model = None
        self.feature_processor = feature_processor
        self.interaction_extractor = interaction_extractor
        
        # 特征名称（用于可解释性）
        self.feature_names = None
        self.embedding_dim = None
        
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
    
    def set_feature_extractors(self, feature_processor, interaction_extractor):
        """设置特征处理器（可以延迟设置）"""
        self.feature_processor = feature_processor
        self.interaction_extractor = interaction_extractor
        self.din_extractor.set_feature_extractors(feature_processor, interaction_extractor)
    
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
        print("训练混合精排模型 (v2)")
        print("=" * 60)
        
        # 1. 提取训练集的 DIN 嵌入
        print("\n1. 提取训练集 DIN 嵌入...")
        train_emb, train_din_scores, train_labels, train_extra = \
            self.din_extractor.extract_embeddings(train_loader, include_model_score=self.use_din_score)
        
        # 2. 提取验证集的 DIN 嵌入
        print("\n2. 提取验证集 DIN 嵌入...")
        valid_emb, valid_din_scores, valid_labels, valid_extra = \
            self.din_extractor.extract_embeddings(valid_loader, include_model_score=self.use_din_score)
        
        # 保存embedding维度
        self.embedding_dim = train_emb.shape[1]
        
        # 3. 构建 LightGBM 特征
        print("\n3. 构建 LightGBM 特征...")
        train_features = self._build_lgb_features(train_emb, train_din_scores, train_extra)
        valid_features = self._build_lgb_features(valid_emb, valid_din_scores, valid_extra)
        
        # 构建特征名称
        self._build_feature_names()
        
        print(f"   训练集: {train_features.shape}")
        print(f"   验证集: {valid_features.shape}")
        print(f"   特征维度: DIN嵌入({self.embedding_dim}) + DIN分数(1) + 显式特征({len(EXPLICIT_FEATURE_NAMES)})")
        
        # 4. 训练 LightGBM
        print("\n4. 训练 LightGBM...")
        train_data = lgb.Dataset(train_features, label=train_labels, feature_name=self.feature_names)
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
        评估混合精排模型（CTR指标）
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
    
    def evaluate_topk(
        self,
        eval_data,
        feature_processor,
        interaction_extractor,
        max_seq_length,
        ks=(5, 10, 20),
        device='cpu'
    ):
        """
        Top-K 推荐评估
        
        Args:
            eval_data: list of dict，每个包含 user_id, history, ground_truth, candidates
            feature_processor: 特征处理器
            interaction_extractor: 交互特征提取器
            max_seq_length: 最大序列长度
            ks: 评估的K值
            device: 设备
        
        Returns:
            dict: HR@K, NDCG@K, MRR@K, Precision@K
        """
        if self.lgb_model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        # 确保特征处理器已设置
        if feature_processor is not None:
            self.din_extractor.set_feature_extractors(feature_processor, interaction_extractor)
        
        from data_loader import build_topk_batch_multi
        
        # 初始化指标
        all_hr = {k: [] for k in ks}
        all_ndcg = {k: [] for k in ks}
        all_mrr = {k: [] for k in ks}
        
        self.din_extractor.din_model.eval()
        
        with torch.no_grad():
            for eval_item in tqdm(eval_data, desc='Hybrid Top-K Eval'):
                # 构建候选batch
                batch = build_topk_batch_multi(
                    eval_item, feature_processor, interaction_extractor,
                    max_seq_length, device
                )
                
                # 提取DIN嵌入和分数
                emb, extra = self.din_extractor._extract_single_batch(batch)
                emb = emb.cpu().numpy()
                
                # DIN分数
                din_logits = self.din_extractor.din_model(batch)
                din_scores = torch.sigmoid(din_logits).cpu().numpy()
                
                # 构建LightGBM特征
                features = self._build_lgb_features(emb, din_scores, extra)
                
                # LightGBM预测
                scores = self.lgb_model.predict(features)
                
                # 排序
                candidates = eval_item['candidates']
                ground_truth = eval_item['ground_truth']
                sorted_indices = np.argsort(-scores)
                ranked_items = [candidates[i] for i in sorted_indices]
                
                # 计算指标
                for k in ks:
                    all_hr[k].append(self._hit_at_k(ranked_items, ground_truth, k))
                    all_ndcg[k].append(self._ndcg_at_k(ranked_items, ground_truth, k))
                    all_mrr[k].append(self._mrr_at_k(ranked_items, ground_truth, k))
        
        # 计算平均值
        results = {}
        for k in ks:
            results[f'HR@{k}'] = np.mean(all_hr[k])
            results[f'NDCG@{k}'] = np.mean(all_ndcg[k])
            results[f'MRR@{k}'] = np.mean(all_mrr[k])
            results[f'Precision@{k}'] = np.mean(all_hr[k]) / k
        
        return results
    
    @staticmethod
    def _hit_at_k(ranked_items, ground_truth, k):
        """计算 Hit@K"""
        return 1.0 if ground_truth in ranked_items[:k] else 0.0
    
    @staticmethod
    def _ndcg_at_k(ranked_items, ground_truth, k):
        """计算 NDCG@K"""
        for i, item in enumerate(ranked_items[:k]):
            if item == ground_truth:
                return 1.0 / np.log2(i + 2)
        return 0.0
    
    @staticmethod
    def _mrr_at_k(ranked_items, ground_truth, k):
        """计算 MRR@K"""
        for i, item in enumerate(ranked_items[:k]):
            if item == ground_truth:
                return 1.0 / (i + 1)
        return 0.0
    
    def _build_feature_names(self):
        """构建特征名称列表"""
        self.feature_names = []
        
        # DIN嵌入特征
        for i in range(self.embedding_dim):
            self.feature_names.append(f'din_emb_{i}')
        
        # DIN分数
        if self.use_din_score:
            self.feature_names.append('din_score')
        
        # 显式特征（与prepare_lightgbm_features对齐）
        self.feature_names.extend(EXPLICIT_FEATURE_NAMES)
    
    def _build_lgb_features(self, embeddings, din_scores, extra_features):
        """
        构建 LightGBM 输入特征
        
        特征组成:
        - DIN embedding（64维用户兴趣向量）
        - DIN 预测分数（1维）
        - 显式特征（20维，与prepare_lightgbm_features对齐）
        
        总计: 64 + 1 + 20 = 85 维
        """
        features_list = [embeddings]
        
        if din_scores is not None:
            features_list.append(din_scores.reshape(-1, 1))
        
        features_list.append(extra_features)
        
        return np.hstack(features_list)
    
    def get_feature_importance(self, top_n=20):
        """获取特征重要性（带有语义化名称）"""
        if self.lgb_model is None:
            raise RuntimeError("模型未训练")
        
        importance = self.lgb_model.feature_importance(importance_type='gain')
        
        # 使用预构建的特征名称
        if self.feature_names is None:
            self._build_feature_names()
        
        # 排序
        indices = np.argsort(importance)[::-1][:top_n]
        
        return [(self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}', 
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


# ========================================
# LightGBM 单独评估器（用于纯LightGBM的Top-K评估）
# ========================================

class LightGBMRanker:
    """
    纯 LightGBM 精排器
    
    用于实验对比，提供与 HybridRanker 一致的 Top-K 评估接口。
    """
    
    def __init__(self, lgb_model, feature_processor, interaction_extractor):
        """
        Args:
            lgb_model: 训练好的 LightGBM 模型
            feature_processor: 特征处理器
            interaction_extractor: 交互特征提取器
        """
        self.lgb_model = lgb_model
        self.feature_processor = feature_processor
        self.interaction_extractor = interaction_extractor
    
    def evaluate_topk(
        self,
        eval_data,
        max_seq_length,
        ks=(5, 10, 20),
        show_progress=True
    ):
        """
        Top-K 推荐评估
        
        Args:
            eval_data: list of dict，每个包含 user_id, history, ground_truth, candidates
            max_seq_length: 最大序列长度
            ks: 评估的K值
            show_progress: 是否显示进度条
        
        Returns:
            dict: HR@K, NDCG@K, MRR@K, Precision@K
        """
        # 初始化指标
        all_hr = {k: [] for k in ks}
        all_ndcg = {k: [] for k in ks}
        all_mrr = {k: [] for k in ks}
        
        iterator = tqdm(eval_data, desc='LightGBM Top-K Eval') if show_progress else eval_data
        for eval_item in iterator:
            user_id = eval_item['user_id']
            history = eval_item['history'][-max_seq_length:]
            candidates = eval_item['candidates']
            ground_truth = eval_item['ground_truth']
            
            # 构建每个候选物品的特征
            features = []
            for item_id in candidates:
                feat = self._build_single_feature(user_id, item_id, history)
                features.append(feat)
            
            features = np.array(features)
            
            # 预测
            scores = self.lgb_model.predict(features)
            
            # 排序
            sorted_indices = np.argsort(-scores)
            ranked_items = [candidates[i] for i in sorted_indices]
            
            # 计算指标
            for k in ks:
                all_hr[k].append(self._hit_at_k(ranked_items, ground_truth, k))
                all_ndcg[k].append(self._ndcg_at_k(ranked_items, ground_truth, k))
                all_mrr[k].append(self._mrr_at_k(ranked_items, ground_truth, k))
        
        # 计算平均值
        results = {}
        for k in ks:
            hr_k = np.mean(all_hr[k])
            results[f'HR@{k}'] = hr_k
            results[f'Recall@{k}'] = hr_k  # 单 ground truth 时 Recall = HR
            results[f'NDCG@{k}'] = np.mean(all_ndcg[k])
            results[f'MRR@{k}'] = np.mean(all_mrr[k])
            results[f'Precision@{k}'] = hr_k / k
        
        return results
    
    def _build_single_feature(self, user_id, item_id, history):
        """构建单个样本的特征向量（与prepare_lightgbm_features对齐）"""
        
        # 用户特征
        user_feat = self.feature_processor.get_user_features(user_id)
        user_activity = self.interaction_extractor.get_user_activity(user_id)
        
        # 物品特征
        item_feat = self.feature_processor.get_item_features(item_id)
        item_popularity = self.interaction_extractor.get_item_popularity(item_id)
        
        # 历史物品特征
        hist_item_feats = [self.feature_processor.get_item_features(iid) for iid in history if iid > 0]
        hist_genres = [f['primary_genre'] for f in hist_item_feats]
        hist_years = [f['year_bucket'] for f in hist_item_feats]
        
        # 序列特征
        seq_length = len(history)
        unique_items = len(set(history))
        avg_popularity = np.mean([self.interaction_extractor.get_item_popularity(iid) for iid in history]) if history else 0
        
        # 构建特征向量（20维）
        sample = [
            user_feat.get('age_bucket', 0),
            user_feat.get('gender', 0),
            user_feat.get('occupation', 0),
            user_activity,
            item_feat.get('year_bucket', 0),
            item_feat.get('primary_genre', 0),
            item_feat.get('genre_count', 1),
            item_popularity,
            seq_length,
            unique_items,
            avg_popularity,
            0.5,  # recency_weight
            0, 0, 0, 0.5,  # 时间特征（简化）
            1 if item_feat.get('primary_genre', 0) in hist_genres else 0,
            1 if item_feat.get('year_bucket', 0) in hist_years else 0,
            np.mean(hist_genres) if hist_genres else 0,
            np.std(hist_genres) if len(hist_genres) > 1 else 0,
        ]
        
        return sample
    
    @staticmethod
    def _hit_at_k(ranked_items, ground_truth, k):
        return 1.0 if ground_truth in ranked_items[:k] else 0.0
    
    @staticmethod
    def _ndcg_at_k(ranked_items, ground_truth, k):
        for i, item in enumerate(ranked_items[:k]):
            if item == ground_truth:
                return 1.0 / np.log2(i + 2)
        return 0.0
    
    @staticmethod
    def _mrr_at_k(ranked_items, ground_truth, k):
        for i, item in enumerate(ranked_items[:k]):
            if item == ground_truth:
                return 1.0 / (i + 1)
        return 0.0


def train_hybrid_ranker(
    din_model,
    train_loader,
    valid_loader,
    test_loader,
    device='cpu',
    feature_processor=None,
    interaction_extractor=None
):
    """
    便捷函数：训练混合精排模型并评估
    
    v2更新：支持特征处理器参数
    """
    print("\n" + "=" * 80)
    print("混合精排: DIN + LightGBM (v2)")
    print("=" * 80)
    
    # 创建混合精排器
    ranker = HybridRanker(
        din_model, 
        device=device,
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
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
    print("混合精排模块测试 (v2)")
    print("=" * 60)
    
    if not HAS_LIGHTGBM:
        print("请安装 LightGBM: pip install lightgbm")
    else:
        print("LightGBM 已安装，混合精排功能可用")
        print("\n使用示例:")
        print("  from hybrid_ranker import HybridRanker, train_hybrid_ranker, LightGBMRanker")
        print("  ")
        print("  # 创建混合精排器（带特征处理器）")
        print("  ranker = HybridRanker(din_model, device='cuda',")
        print("                        feature_processor=fp, interaction_extractor=ie)")
        print("  ranker.fit(train_loader, valid_loader)")
        print("  ")
        print("  # CTR评估")
        print("  results = ranker.evaluate(test_loader)")
        print("  ")
        print("  # Top-K评估")
        print("  topk_results = ranker.evaluate_topk(eval_data, fp, ie, max_seq_len)")
        print("  ")
        print("  # 纯LightGBM的Top-K评估")
        print("  lgb_ranker = LightGBMRanker(lgb_model, fp, ie)")
        print("  lgb_topk = lgb_ranker.evaluate_topk(eval_data, max_seq_len)")
