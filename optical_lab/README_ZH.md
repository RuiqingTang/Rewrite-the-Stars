# StarNet
## 环境搭建
1. 创建激活虚拟环境（conda）
```
conda create -n starnet python=3.10
conda activate startnet
```
2. 安装依赖项
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install timm
pip install numpy==1.23.3
pip install matplotlib
```
## 训练模型
```
python train.py
```
