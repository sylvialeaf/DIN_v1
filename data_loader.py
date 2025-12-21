#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版数据加载器

支持丰富的特征：
- 用户特征：年龄、性别、职业
- 物品特征：类型、年份
- 交互特征：时间、统计
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
import urllib.request
import zipfile

from feature_engineering import FeatureProcessor, InteractionFeatureExtractor, GENRES


class RichFeatureDataset(Dataset):
    """
    增强版数据集
    
    包含完整的用户、物品、序列特征。
    """
    
    def __init__(
        self,
        data_dir,
        dataset_name='ml-100k',
        max_seq_length=50,
        min_seq_length=5,
        split='train',
        train_ratio=0.8,
        valid_ratio=0.1,
        feature_processor=None,
        interaction_extractor=None
    ):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.split = split
        
        # 加载原始数据
        self._download_if_needed()
        self.raw_data = self._load_raw_data()
        
        # 特征处理器
        if feature_processor is None:
            self.feature_processor = FeatureProcessor(data_dir, dataset_name)
        else:
            self.feature_processor = feature_processor
        
        # 交互特征提取器
        if interaction_extractor is None:
            self.interaction_extractor = InteractionFeatureExtractor(self.raw_data)
        else:
            self.interaction_extractor = interaction_extractor
        
        # 构建用户历史序列
        self.user_sequences = self._build_sequences()
        
        # 划分数据集
        self._split_data(train_ratio, valid_ratio)
        
        # 构建样本
        self.samples = self._build_samples()
    
    def _download_if_needed(self):
        """如果数据不存在，自动下载"""
        data_path = os.path.join(self.data_dir, self.dataset_name)
        
        if os.path.exists(data_path):
            return
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        if self.dataset_name == 'ml-100k':
            url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
        elif self.dataset_name == 'ml-1m':
            url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        print(f"下载数据集 {self.dataset_name}...")
        zip_path = os.path.join(self.data_dir, f'{self.dataset_name}.zip')
        urllib.request.urlretrieve(url, zip_path)
        
        print("解压数据...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        os.remove(zip_path)
        print("数据准备完成!")
    
    def _load_raw_data(self):
        """加载原始评分数据"""
        data_path = os.path.join(self.data_dir, self.dataset_name)
        
        if self.dataset_name == 'ml-100k':
            file_path = os.path.join(data_path, 'u.data')
            df = pd.read_csv(
                file_path, 
                sep='\t', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
        elif self.dataset_name == 'ml-1m':
            file_path = os.path.join(data_path, 'ratings.dat')
            df = pd.read_csv(
                file_path, 
                sep='::', 
                names=['user_id', 'item_id', 'rating', 'timestamp'],
                engine='python'
            )
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        self.num_users = df['user_id'].nunique()
        self.num_items = df['item_id'].nunique()
        
        print(f"数据集: {self.dataset_name}")
        print(f"  用户数: {self.num_users}")
        print(f"  物品数: {self.num_items}")
        print(f"  交互数: {len(df)}")
        
        return df
    
    def _build_sequences(self):
        """按用户构建时间排序的交互序列（优化版本）"""
        user_sequences = defaultdict(list)
        
        # 排序
        sorted_data = self.raw_data.sort_values(['user_id', 'timestamp'])
        
        # 使用 groupby 代替 iterrows，速度快 10-100 倍
        for user_id, group in sorted_data.groupby('user_id', sort=False):
            items = group['item_id'].tolist()
            ratings = group['rating'].tolist()
            timestamps = group['timestamp'].tolist()
            
            for i in range(len(items)):
                user_sequences[user_id].append({
                    'item_id': items[i],
                    'rating': ratings[i],
                    'timestamp': timestamps[i]
                })
        
        filtered_sequences = {
            u: seq for u, seq in user_sequences.items() 
            if len(seq) >= self.min_seq_length
        }
        
        print(f"  有效用户数（序列长度 >= {self.min_seq_length}）: {len(filtered_sequences)}")
        
        return filtered_sequences
    
    def _split_data(self, train_ratio, valid_ratio):
        """按用户划分训练/验证/测试集"""
        all_users = list(self.user_sequences.keys())
        np.random.seed(2020)
        np.random.shuffle(all_users)
        
        n_train = int(len(all_users) * train_ratio)
        n_valid = int(len(all_users) * valid_ratio)
        
        if self.split == 'train':
            self.active_users = all_users[:n_train]
        elif self.split == 'valid':
            self.active_users = all_users[n_train:n_train + n_valid]
        elif self.split == 'test':
            self.active_users = all_users[n_train + n_valid:]
        else:
            raise ValueError(f"无效的 split: {self.split}")
    
    def _build_samples(self):
        """构建训练样本，包含丰富特征"""
        samples = []
        
        for user_id in self.active_users:
            seq = self.user_sequences[user_id]
            
            if len(seq) < 2:
                continue
            
            items = [s['item_id'] for s in seq]
            timestamps = [s['timestamp'] for s in seq]
            
            # 获取用户特征
            user_feat = self.feature_processor.get_user_features(user_id)
            
            for i in range(1, len(items)):
                history = items[:i]
                history_ts = timestamps[:i]
                
                if len(history) > self.max_seq_length:
                    history = history[-self.max_seq_length:]
                    history_ts = history_ts[-self.max_seq_length:]
                
                positive_item = items[i]
                
                # 正样本
                samples.append({
                    'user_id': user_id,
                    'history': history,
                    'history_timestamps': history_ts,
                    'target_item': positive_item,
                    'timestamp': timestamps[i],
                    'label': 1,
                    **user_feat
                })
                
                # 负样本
                user_items = set(items)
                negative_item = np.random.randint(1, self.num_items + 1)
                while negative_item in user_items:
                    negative_item = np.random.randint(1, self.num_items + 1)
                
                samples.append({
                    'user_id': user_id,
                    'history': history,
                    'history_timestamps': history_ts,
                    'target_item': negative_item,
                    'timestamp': timestamps[i],
                    'label': 0,
                    **user_feat
                })
        
        print(f"  {self.split} 样本数: {len(samples)}")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        history = sample['history']
        seq_len = len(history)
        
        # 获取历史物品特征
        history_genres = []
        history_years = []
        for item_id in history:
            item_feat = self.feature_processor.get_item_features(item_id)
            history_genres.append(item_feat['primary_genre'])
            history_years.append(item_feat['year_bucket'])
        
        # Padding
        if len(history) < self.max_seq_length:
            padding_len = self.max_seq_length - len(history)
            history = [0] * padding_len + history
            history_genres = [0] * padding_len + history_genres
            history_years = [0] * padding_len + history_years
            mask = [0] * padding_len + [1] * seq_len
        else:
            mask = [1] * self.max_seq_length
        
        # 目标物品特征
        target_item = sample['target_item']
        target_feat = self.feature_processor.get_item_features(target_item)
        
        # 时间特征
        time_feat = self.interaction_extractor.get_time_features(sample['timestamp'])
        
        # 用户统计特征
        user_activity = self.interaction_extractor.get_user_activity(sample['user_id'])
        item_popularity = self.interaction_extractor.get_item_popularity(target_item)
        
        return {
            # 基础 ID
            'user_id': torch.tensor(sample['user_id'], dtype=torch.long),
            'item_seq': torch.tensor(history, dtype=torch.long),
            'item_seq_mask': torch.tensor(mask, dtype=torch.float),
            'target_item': torch.tensor(target_item, dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.float),
            'seq_len': torch.tensor(seq_len, dtype=torch.long),
            
            # 用户特征
            'user_age': torch.tensor(sample['age_bucket'], dtype=torch.long),
            'user_gender': torch.tensor(sample['gender'], dtype=torch.long),
            'user_occupation': torch.tensor(sample['occupation'], dtype=torch.long),
            'user_activity': torch.tensor(user_activity, dtype=torch.long),
            
            # 物品特征
            'item_genre': torch.tensor(target_feat['primary_genre'], dtype=torch.long),
            'item_year': torch.tensor(target_feat['year_bucket'], dtype=torch.long),
            'item_popularity': torch.tensor(item_popularity, dtype=torch.long),
            
            # 历史序列特征
            'history_genres': torch.tensor(history_genres, dtype=torch.long),
            'history_years': torch.tensor(history_years, dtype=torch.long),
            
            # 时间特征
            'time_hour': torch.tensor(time_feat['hour_bucket'], dtype=torch.long),
            'time_dow': torch.tensor(time_feat['day_of_week'], dtype=torch.long),
            'time_weekend': torch.tensor(time_feat['is_weekend'], dtype=torch.long),
        }


def get_rich_dataloaders(
    data_dir='./data',
    dataset_name='ml-100k',
    max_seq_length=50,
    batch_size=256,
    num_workers=0
):
    """
    获取带丰富特征的数据加载器
    
    Returns:
        train_loader, valid_loader, test_loader, dataset_info, feature_processor
    """
    
    # 共享特征处理器
    feature_processor = FeatureProcessor(data_dir, dataset_name)
    
    # 加载交互数据用于统计
    data_path = os.path.join(data_dir, dataset_name)
    if dataset_name == 'ml-100k':
        interactions = pd.read_csv(
            os.path.join(data_path, 'u.data'),
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp']
        )
    else:
        interactions = pd.read_csv(
            os.path.join(data_path, 'ratings.dat'),
            sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
    
    interaction_extractor = InteractionFeatureExtractor(interactions)
    
    # 创建数据集
    train_dataset = RichFeatureDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split='train',
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    valid_dataset = RichFeatureDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split='valid',
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    test_dataset = RichFeatureDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
        split='test',
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    # 特征维度信息
    feature_dims = feature_processor.get_feature_dims()
    feature_dims['num_users'] = train_dataset.num_users
    feature_dims['num_items'] = train_dataset.num_items
    feature_dims['time_hour'] = 7  # 0-6
    feature_dims['time_dow'] = 8   # 0-7
    feature_dims['time_weekend'] = 2
    feature_dims['user_activity'] = 6
    feature_dims['item_popularity'] = 6
    
    dataset_info = {
        'num_users': train_dataset.num_users,
        'num_items': train_dataset.num_items,
        'max_seq_length': max_seq_length,
        'feature_dims': feature_dims
    }
    
    return train_loader, valid_loader, test_loader, dataset_info, feature_processor


if __name__ == "__main__":
    print("测试增强版数据加载器...")
    
    train_loader, valid_loader, test_loader, info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name='ml-100k',
        max_seq_length=50,
        batch_size=32
    )
    
    print("\n数据集信息:")
    print(info)
    
    print("\n示例 batch:")
    batch = next(iter(train_loader))
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
