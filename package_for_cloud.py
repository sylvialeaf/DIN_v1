#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
打包项目用于云端部署

生成 standalone_din_package.zip，包含所有必要文件。
"""

import os
import zipfile
from datetime import datetime

# 需要包含的文件
INCLUDE_FILES = [
    'models.py',
    'data_loader.py', 
    'feature_engineering.py',
    'trainer.py',
    'hybrid_ranker.py',
    'experiment1.py',
    'experiment2.py',
    'experiment3.py',
    'run_all_gpu.py',
    'run_experiments.py',
    'README.md',
    'CLOUD_GPU_GUIDE.md',
    'requirements.txt',
]

# 可选包含的目录
INCLUDE_DIRS = []  # 不包含 data，云端会自动下载

def create_package():
    """创建部署包"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_name = f'standalone_din_package_{timestamp}.zip'
    zip_path = os.path.join(os.path.dirname(script_dir), zip_name)
    
    print("📦 正在打包项目...")
    print(f"   目标: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for fname in INCLUDE_FILES:
            fpath = os.path.join(script_dir, fname)
            if os.path.exists(fpath):
                arcname = os.path.join('standalone_din', fname)
                zipf.write(fpath, arcname)
                print(f"   ✅ {fname}")
            else:
                print(f"   ⚠️ 跳过不存在的文件: {fname}")
        
        for dirname in INCLUDE_DIRS:
            dirpath = os.path.join(script_dir, dirname)
            if os.path.exists(dirpath):
                for root, dirs, files in os.walk(dirpath):
                    for f in files:
                        fpath = os.path.join(root, f)
                        arcname = os.path.join('standalone_din', os.path.relpath(fpath, script_dir))
                        zipf.write(fpath, arcname)
                print(f"   ✅ {dirname}/")
    
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n✅ 打包完成!")
    print(f"   文件: {zip_path}")
    print(f"   大小: {size_mb:.2f} MB")
    print(f"\n📤 请将此文件上传到云端服务器")
    
    return zip_path


if __name__ == '__main__':
    create_package()
