#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单独运行混合精排模型 (DIN + LightGBM)

功能：
1. 训练 DIN 模型（如果没有预训练模型）
2. 训练混合精排模型
3. 评估并保存特征重要性等可解释性分析结果

用法：
    python run_hybrid_only.py                    # 默认配置
    python run_hybrid_only.py --dataset ml-1m    # 使用ml-1m数据集
    python run_hybrid_only.py --seq_length 100   # 自定义序列长度
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

print("🔧 导入依赖库...")

try:
    import torch
    print("✅ PyTorch 导入成功")
except ImportError as e:
    print(f"❌ PyTorch 导入失败: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    print("✅ NumPy, Pandas 导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader import MovieLensDataLoader
    from feature_engineering import FeatureProcessor, InteractionExtractor
    from models import DIN
    from trainer import Trainer
    from hybrid_ranker import HybridRanker, EXPLICIT_FEATURE_NAMES
    print("✅ 项目模块导入成功")
except ImportError as e:
    print(f"❌ 项目模块导入失败: {e}")
    print(f"   当前目录: {os.getcwd()}")
    print(f"   脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
    sys.exit(1)

# ========================================
# 配置
# ========================================

def parse_args():
    parser = argparse.ArgumentParser(description='单独运行混合精排模型')
    parser.add_argument('--dataset', type=str, default='ml-100k', choices=['ml-100k', 'ml-1m'],
                        help='数据集名称')
    parser.add_argument('--seq_length', type=int, default=50, help='最大序列长度')
    parser.add_argument('--embedding_dim', type=int, default=64, help='嵌入维度')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--epochs', type=int, default=30, help='DIN训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--lgb_rounds', type=int, default=300, help='LightGBM迭代次数')
    parser.add_argument('--early_stop', type=int, default=30, help='早停轮数')
    parser.add_argument('--device', type=str, default='auto', help='设备 (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='results', help='结果保存目录')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("🚀 开始运行...")
    print(f"📍 参数解析成功: dataset={args.dataset}, seq_length={args.seq_length}")
    
    # 设备
    if args.device == 'auto':
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = args.device
    
    print("=" * 80)
    print("🚀 混合精排模型单独训练脚本")
    print("=" * 80)
    print(f"\n📋 配置:")
    print(f"   数据集: {args.dataset}")
    print(f"   序列长度: {args.seq_length}")
    print(f"   嵌入维度: {args.embedding_dim}")
    print(f"   设备: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    start_time = datetime.now()
    
    # ========================================
    # 1. 数据加载
    # ========================================
    print("\n" + "=" * 60)
    print("📦 Step 1: 加载数据")
    print("=" * 60)
    
    data_loader = MovieLensDataLoader(
        dataset_name=args.dataset,
        max_seq_length=args.seq_length
    )
    
    train_loader, valid_loader, test_loader = data_loader.get_dataloaders(
        batch_size=args.batch_size,
        num_workers=4
    )
    
    dataset_info = data_loader.get_dataset_info()
    print(f"✅ 数据加载完成")
    print(f"   用户数: {dataset_info['num_users']}")
    print(f"   物品数: {dataset_info['num_items']}")
    print(f"   类型数: {dataset_info['num_genres']}")
    print(f"   训练样本: {len(train_loader.dataset)}")
    
    # ========================================
    # 2. 特征处理器
    # ========================================
    print("\n" + "=" * 60)
    print("🔧 Step 2: 初始化特征处理器")
    print("=" * 60)
    
    feature_processor = FeatureProcessor(data_loader.ratings_df, data_loader.users_df, data_loader.items_df)
    interaction_extractor = InteractionExtractor(data_loader.ratings_df)
    
    print(f"✅ 特征处理器初始化完成")
    print(f"   显式特征数: {len(EXPLICIT_FEATURE_NAMES)}")
    
    # ========================================
    # 3. 训练 DIN 模型
    # ========================================
    print("\n" + "=" * 60)
    print("🧠 Step 3: 训练 DIN 模型")
    print("=" * 60)
    
    din_config = {
        'num_users': dataset_info['num_users'],
        'num_items': dataset_info['num_items'],
        'num_genres': dataset_info['num_genres'],
        'num_years': 10,
        'embedding_dim': args.embedding_dim,
        'mlp_dims': [256, 128, 64],
        'dropout': 0.2,
        'use_time_decay': True,
        'time_decay_lambda': 0.1,
    }
    
    din_model = DIN(**din_config).to(DEVICE)
    
    trainer = Trainer(
        model=din_model,
        device=DEVICE,
        learning_rate=args.lr,
        weight_decay=1e-5
    )
    
    t1 = time.time()
    trainer.train(
        train_loader,
        valid_loader,
        epochs=args.epochs,
        early_stopping_patience=10
    )
    din_train_time = time.time() - t1
    
    # DIN 评估
    din_test_results = trainer.evaluate(test_loader)
    print(f"\n✅ DIN 训练完成")
    print(f"   训练时间: {din_train_time:.1f}s")
    print(f"   Test AUC: {din_test_results['auc']:.4f}")
    print(f"   Test LogLoss: {din_test_results['logloss']:.4f}")
    
    # ========================================
    # 4. 训练混合精排模型
    # ========================================
    print("\n" + "=" * 60)
    print("🔥 Step 4: 训练混合精排模型 (DIN + LightGBM)")
    print("=" * 60)
    
    hybrid_ranker = HybridRanker(
        din_model,
        device=DEVICE,
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor
    )
    
    t2 = time.time()
    hybrid_ranker.fit(
        train_loader,
        valid_loader,
        num_boost_round=args.lgb_rounds,
        early_stopping_rounds=args.early_stop
    )
    lgb_train_time = time.time() - t2
    
    # 混合模型评估
    hybrid_test_results = hybrid_ranker.evaluate(test_loader)
    
    print(f"\n✅ 混合精排训练完成")
    print(f"   LightGBM训练时间: {lgb_train_time:.1f}s")
    print(f"   Test AUC: {hybrid_test_results['auc']:.4f}")
    print(f"   Test LogLoss: {hybrid_test_results['logloss']:.4f}")
    
    # ========================================
    # 5. Top-K 评估
    # ========================================
    print("\n" + "=" * 60)
    print("📊 Step 5: Top-K 推荐评估")
    print("=" * 60)
    
    # 准备 Top-K 评估数据
    eval_data = data_loader.prepare_topk_eval_data(num_neg=99)
    
    topk_metrics = hybrid_ranker.evaluate_topk(
        eval_data=eval_data,
        feature_processor=feature_processor,
        interaction_extractor=interaction_extractor,
        max_seq_length=args.seq_length,
        ks=(5, 10, 20),
        device=DEVICE
    )
    
    print(f"✅ Top-K 评估完成")
    for k in [5, 10, 20]:
        print(f"   @{k}: HR={topk_metrics[f'HR@{k}']:.4f}, NDCG={topk_metrics[f'NDCG@{k}']:.4f}, MRR={topk_metrics[f'MRR@{k}']:.4f}")
    
    # ========================================
    # 6. 可解释性分析
    # ========================================
    print("\n" + "=" * 60)
    print("🔍 Step 6: 可解释性分析 - 特征重要性")
    print("=" * 60)
    
    # 获取特征重要性（Top 20）
    feature_importance = hybrid_ranker.get_feature_importance(20)
    feature_importance_dict = {name: float(imp) for name, imp in feature_importance}
    
    print("\n📈 特征重要性 Top 20:")
    print("-" * 50)
    
    # 分类展示
    din_emb_features = []
    explicit_features = []
    
    for i, (name, imp) in enumerate(feature_importance, 1):
        if name.startswith('din_emb_'):
            din_emb_features.append((name, imp))
        else:
            explicit_features.append((name, imp))
        print(f"  {i:2d}. {name:<25s}: {imp:>10.2f}")
    
    # 统计分析
    total_importance = sum(imp for _, imp in feature_importance)
    din_importance = sum(imp for _, imp in din_emb_features)
    explicit_importance = sum(imp for _, imp in explicit_features)
    
    print("\n📊 特征贡献分析:")
    print(f"   DIN嵌入特征贡献: {din_importance/total_importance*100:.1f}%")
    print(f"   显式特征贡献: {explicit_importance/total_importance*100:.1f}%")
    
    if explicit_features:
        print(f"\n📝 最重要的显式特征:")
        for name, imp in explicit_features[:5]:
            print(f"   - {name}: {imp:.2f}")
    
    # ========================================
    # 7. 模型对比
    # ========================================
    print("\n" + "=" * 60)
    print("⚖️ Step 7: DIN vs Hybrid 对比")
    print("=" * 60)
    
    comparison = hybrid_ranker.compare_with_din()
    
    # ========================================
    # 8. 保存结果
    # ========================================
    print("\n" + "=" * 60)
    print("💾 Step 8: 保存结果")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    # 构建完整结果
    results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'config': {
            'dataset': args.dataset,
            'seq_length': args.seq_length,
            'embedding_dim': args.embedding_dim,
            'batch_size': args.batch_size,
            'din_epochs': args.epochs,
            'lgb_rounds': args.lgb_rounds,
            'device': DEVICE,
            'gpu_name': torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'N/A'
        },
        'din_results': {
            'test_auc': din_test_results['auc'],
            'test_logloss': din_test_results['logloss'],
            'train_time_sec': din_train_time,
            'num_params': sum(p.numel() for p in din_model.parameters())
        },
        'hybrid_results': {
            'test_auc': hybrid_test_results['auc'],
            'test_logloss': hybrid_test_results['logloss'],
            'lgb_train_time_sec': lgb_train_time,
            'total_train_time_sec': din_train_time + lgb_train_time,
            'lgb_num_trees': hybrid_ranker.lgb_model.num_trees(),
            'lgb_best_iteration': hybrid_ranker.lgb_model.best_iteration
        },
        'topk_metrics': topk_metrics,
        'interpretability': {
            'feature_importance_top20': feature_importance_dict,
            'embedding_dim': hybrid_ranker.embedding_dim,
            'num_explicit_features': len(EXPLICIT_FEATURE_NAMES),
            'total_features': len(hybrid_ranker.feature_names),
            'din_emb_contribution_pct': din_importance / total_importance * 100,
            'explicit_contribution_pct': explicit_importance / total_importance * 100,
            'top_explicit_features': {name: float(imp) for name, imp in explicit_features[:10]}
        },
        'comparison': {
            'din_auc': comparison['din']['auc'],
            'hybrid_auc': comparison['hybrid']['auc'],
            'auc_improvement_pct': comparison['auc_improvement']
        },
        'total_time_minutes': total_time / 60
    }
    
    # 保存 JSON
    json_file = os.path.join(args.output_dir, f'hybrid_analysis_{args.dataset}.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到: {json_file}")
    
    # ========================================
    # 9. 总结
    # ========================================
    print("\n" + "=" * 80)
    print("🎉 训练完成!")
    print("=" * 80)
    print(f"\n📊 性能总结:")
    print(f"   {'模型':<15} {'AUC':<12} {'LogLoss':<12}")
    print(f"   {'-'*40}")
    print(f"   {'DIN':<15} {din_test_results['auc']:<12.4f} {din_test_results['logloss']:<12.4f}")
    print(f"   {'Hybrid':<15} {hybrid_test_results['auc']:<12.4f} {hybrid_test_results['logloss']:<12.4f}")
    print(f"   {'-'*40}")
    print(f"   AUC变化: {comparison['auc_improvement']:+.2f}%")
    
    print(f"\n⏱️ 时间统计:")
    print(f"   DIN训练: {din_train_time:.1f}s")
    print(f"   LightGBM训练: {lgb_train_time:.1f}s")
    print(f"   总时间: {total_time/60:.1f}分钟")
    
    print(f"\n📁 输出文件:")
    print(f"   {json_file}")
    
    return results


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🎬 混合精排模型训练脚本启动")
    print("=" * 80)
    
    try:
        results = main()
        print("\n✅ 脚本执行成功!")
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print("\n\n❌ 脚本执行失败!")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
