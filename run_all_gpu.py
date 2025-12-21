#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
云端 GPU 完整实验脚本

适合在 AutoDL / Colab / 阿里云等 GPU 环境运行。
支持 ml-100k 和 ml-1m 双数据集。

使用方法:
    python run_all_gpu.py                    # 运行所有实验（两个数据集）
    python run_all_gpu.py --dataset ml-100k  # 只运行 ml-100k
    python run_all_gpu.py --dataset ml-1m    # 只运行 ml-1m
    python run_all_gpu.py --quick            # 快速测试模式

预估时间 (GPU):
    ml-100k: 约 15-20 分钟
    ml-1m:   约 60-90 分钟
    总计:    约 1.5-2 小时
"""

import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

from data_loader import get_rich_dataloaders
from models import DINRichLite, SimpleAveragePoolingRich, GRU4Rec, SASRec, NARM
from trainer import RichTrainer, measure_inference_speed_rich
from feature_engineering import FeatureProcessor, InteractionFeatureExtractor, prepare_lightgbm_features

# ========================================
# 配置
# ========================================

# 解析命令行参数
parser = argparse.ArgumentParser(description='云端 GPU 完整实验')
parser.add_argument('--dataset', type=str, default='both', 
                    choices=['ml-100k', 'ml-1m', 'both'],
                    help='数据集选择')
parser.add_argument('--quick', action='store_true', 
                    help='快速测试模式（减少 epochs 和序列长度）')
parser.add_argument('--epochs', type=int, default=50,
                    help='训练轮数（默认 50）')
args = parser.parse_args()

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
print(f"数据集: {DATASETS}")
print(f"Epochs: {EPOCHS}")
print(f"序列长度: {SEQ_LENGTHS}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"模型: {MODELS_TO_TEST}")
print(f"快速模式: {args.quick}")
print("=" * 80)

all_results = []
experiment_start = datetime.now()


# ========================================
# 实验一：序列长度敏感性 + 模型对比
# ========================================

def run_experiment1(dataset_name):
    """实验一：不同序列长度下各模型的表现"""
    print("\n" + "=" * 80)
    print(f"📊 实验一：序列长度敏感性 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    
    for seq_length in SEQ_LENGTHS:
        print(f"\n🔬 序列长度: {seq_length}")
        
        # 加载数据
        train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
            data_dir='./data',
            dataset_name=dataset_name,
            max_seq_length=seq_length,
            batch_size=BATCH_SIZE
        )
        
        for model_name in MODELS_TO_TEST:
            print(f"  🚀 {model_name}...", end=" ", flush=True)
            
            try:
                # 创建模型
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
                
                # 训练
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
                
                # 评估
                test_metrics = trainer.evaluate(test_loader)
                speed = measure_inference_speed_rich(model, test_loader, DEVICE)
                
                results.append({
                    'experiment': 'exp1',
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
                    'experiment': 'exp1',
                    'dataset': dataset_name,
                    'seq_length': seq_length,
                    'model': model_name,
                    'test_auc': None,
                    'status': f'error: {str(e)[:100]}'
                })
    
    return results


# ========================================
# 实验二：方法对比 + LightGBM
# ========================================

def run_experiment2(dataset_name):
    """实验二：DIN vs 传统方法"""
    print("\n" + "=" * 80)
    print(f"📊 实验二：方法对比 [{dataset_name}]")
    print("=" * 80)
    
    results = []
    seq_length = 50
    
    # 加载数据
    train_loader, valid_loader, test_loader, dataset_info, fp = get_rich_dataloaders(
        data_dir='./data',
        dataset_name=dataset_name,
        max_seq_length=seq_length,
        batch_size=BATCH_SIZE
    )
    
    # 测试各模型
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
                'experiment': 'exp2',
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
                'experiment': 'exp2',
                'dataset': dataset_name,
                'model': model_name,
                'test_auc': None,
                'status': f'error: {str(e)[:100]}'
            })
    
    # LightGBM
    print("  🚀 LightGBM...", end=" ", flush=True)
    try:
        import lightgbm as lgb
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
            'experiment': 'exp2',
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
            'experiment': 'exp2',
            'dataset': dataset_name,
            'model': 'LightGBM',
            'test_auc': None,
            'status': f'error: {str(e)[:100]}'
        })
    
    return results


# ========================================
# 主程序
# ========================================

if __name__ == '__main__':
    print(f"\n⏰ 实验开始时间: {experiment_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for dataset in DATASETS:
        print(f"\n{'='*80}")
        print(f"📁 数据集: {dataset.upper()}")
        print(f"{'='*80}")
        
        # 运行实验一
        results1 = run_experiment1(dataset)
        all_results.extend(results1)
        
        # 运行实验二
        results2 = run_experiment2(dataset)
        all_results.extend(results2)
    
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
        'datasets': DATASETS,
        'epochs': EPOCHS,
        'seq_lengths': SEQ_LENGTHS,
        'models': MODELS_TO_TEST,
        'total_time_minutes': total_time / 60,
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
    print(f"\n结果文件:")
    print(f"  - {csv_file}")
    print(f"  - {json_file}")
    
    # 各数据集最佳结果
    print("\n📊 各数据集最佳 AUC:")
    df_success = df_results[df_results['status'] == 'success']
    for dataset in DATASETS:
        df_ds = df_success[df_success['dataset'] == dataset]
        if len(df_ds) > 0:
            best = df_ds.loc[df_ds['test_auc'].idxmax()]
            print(f"  {dataset}: {best['model']} = {best['test_auc']:.4f}")
    
    print("=" * 80)
