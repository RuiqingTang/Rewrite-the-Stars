#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data_gen.py
@Time    :   2025/04/03 09:51:35
@Author  :   Ruiqing Tang 
@Contact :   tangruiqing123@gmail.com
'''

import csv
import random

# 设置随机种子以确保结果可重复
random.seed(42)

# 创建CSV文件
with open('synthetic_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # 写入表头
    header = [f'feature_{i+1}' for i in range(16)] + ['class']
    writer.writerow(header)
    
    # 生成500个样本
    for _ in range(500):
        # 生成16个随机特征值
        features = [round(random.uniform(0, 1), 4) for _ in range(16)]
        
        # 根据特征值的总和决定类别（0-5）
        total = sum(features)
        if total < 8:
            class_label = 0
        elif total < 9:
            class_label = 1
        elif total < 10:
            class_label = 2
        elif total < 11:
            class_label = 3
        elif total < 12:
            class_label = 4
        else:
            class_label = 5
        
        # 写入一行数据
        writer.writerow(features + [class_label])
        
print("数据集已成功生成并保存为 'synthetic_dataset.csv'")