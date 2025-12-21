#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
增强版训练器

支持丰富特征的模型训练。
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
from tqdm import tqdm
import time


class RichTrainer:
    """
    增强版训练器
    
    支持 batch 字典形式的输入。
    """
    
    def __init__(
        self,
        model,
        device='cpu',
        learning_rate=1e-3,
        weight_decay=1e-5
    ):
        self.model = model.to(device)
        self.device = device
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
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
    
    from data_loader_rich import get_rich_dataloaders
    from models_rich import DINRichLite
    
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
