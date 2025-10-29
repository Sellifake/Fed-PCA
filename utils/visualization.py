"""
Visualization Utilities for Fed-ProFiLA-AD
实现训练过程可视化和结果分析
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def plot_training_curves(training_history: Dict, save_path: str = "results/training_curves.png") -> None:
    """
    绘制训练曲线
    
    Args:
        training_history: 训练历史字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fed-ProFiLA-AD Training Progress', fontsize=16)
    
    # 1. 损失曲线
    if 'rounds' in training_history and 'global_losses' in training_history:
        axes[0, 0].plot(training_history['rounds'], training_history['global_losses'], 'b-', label='Global Loss')
        axes[0, 0].set_xlabel('Federation Round')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Global Loss Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # 2. 原型距离
    if 'rounds' in training_history and 'prototype_distances' in training_history:
        axes[0, 1].plot(training_history['rounds'], training_history['prototype_distances'], 'r-', label='Prototype Distance')
        axes[0, 1].set_xlabel('Federation Round')
        axes[0, 1].set_ylabel('Distance')
        axes[0, 1].set_title('Prototype Distance Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # 3. 评估指标
    if 'rounds' in training_history and 'eval_aucs' in training_history:
        axes[1, 0].plot(training_history['rounds'], training_history['eval_aucs'], 'g-', label='AUC')
        axes[1, 0].set_xlabel('Federation Round')
        axes[1, 0].set_ylabel('AUC Score')
        axes[1, 0].set_title('Evaluation AUC Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # 4. 客户端参与情况
    if 'selected_clients' in training_history:
        client_participation = {}
        for round_clients in training_history['selected_clients']:
            for client_id in round_clients:
                client_participation[client_id] = client_participation.get(client_id, 0) + 1
        
        clients = list(client_participation.keys())
        participation_counts = list(client_participation.values())
        
        axes[1, 1].bar(clients, participation_counts, color='skyblue')
        axes[1, 1].set_xlabel('Client ID')
        axes[1, 1].set_ylabel('Participation Count')
        axes[1, 1].set_title('Client Participation')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training curves saved to {save_path}")


def plot_client_metrics(eval_results: Dict, save_path: str = "results/client_metrics.png") -> None:
    """
    绘制客户端评估指标
    
    Args:
        eval_results: 评估结果字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 提取数据
    clients = []
    metrics = ['auc', 'f1', 'precision', 'recall']
    metric_data = {metric: [] for metric in metrics}
    
    for client_id, result in eval_results.items():
        if client_id == 'average':
            continue
        clients.append(client_id)
        for metric in metrics:
            metric_data[metric].append(result.get(metric, 0))
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Client Evaluation Metrics', fontsize=16)
    
    for i, metric in enumerate(metrics):
        row = i // 2
        col = i % 2
        
        axes[row, col].bar(clients, metric_data[metric], color=plt.cm.viridis(np.linspace(0, 1, len(clients))))
        axes[row, col].set_xlabel('Client ID')
        axes[row, col].set_ylabel(metric.upper())
        axes[row, col].set_title(f'{metric.upper()} by Client')
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True, alpha=0.3)
        
        # 添加数值标签
        for j, v in enumerate(metric_data[metric]):
            axes[row, col].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Client metrics plot saved to {save_path}")


def plot_loss_distribution(client_losses: Dict, save_path: str = "results/loss_distribution.png") -> None:
    """
    绘制损失分布
    
    Args:
        client_losses: 客户端损失字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Loss Distribution Analysis', fontsize=16)
    
    # 1. 总损失分布
    all_losses = []
    for client_id, losses in client_losses.items():
        all_losses.extend(losses)
    
    axes[0, 0].hist(all_losses, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Loss Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Loss Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 各客户端损失对比
    client_names = list(client_losses.keys())
    client_avg_losses = [np.mean(losses) for losses in client_losses.values()]
    
    axes[0, 1].bar(client_names, client_avg_losses, color=plt.cm.Set3(np.linspace(0, 1, len(client_names))))
    axes[0, 1].set_xlabel('Client ID')
    axes[0, 1].set_ylabel('Average Loss')
    axes[0, 1].set_title('Average Loss by Client')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 损失趋势
    for i, (client_id, losses) in enumerate(client_losses.items()):
        axes[1, 0].plot(losses, label=client_id, alpha=0.7)
    axes[1, 0].set_xlabel('Training Step')
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].set_title('Loss Trends by Client')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 箱线图
    axes[1, 1].boxplot(client_losses.values(), labels=client_names)
    axes[1, 1].set_xlabel('Client ID')
    axes[1, 1].set_ylabel('Loss Value')
    axes[1, 1].set_title('Loss Distribution by Client')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss distribution plot saved to {save_path}")


def create_training_summary(training_results: Dict, eval_results: Dict, save_path: str = "results/training_summary.txt") -> None:
    """
    创建训练摘要报告
    
    Args:
        training_results: 训练结果
        eval_results: 评估结果
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("Fed-ProFiLA-AD Training Summary Report\n")
        f.write("="*60 + "\n\n")
        
        # 训练基本信息
        f.write("Training Configuration:\n")
        f.write("-" * 30 + "\n")
        if 'training_history' in training_results:
            history = training_results['training_history']
            f.write(f"Total Rounds: {len(history.get('rounds', []))}\n")
            f.write(f"Final Prototype Distance: {history.get('prototype_distances', [0])[-1]:.4f}\n")
        
        # 客户端参与情况
        f.write("\nClient Participation:\n")
        f.write("-" * 30 + "\n")
        if 'training_history' in training_results and 'selected_clients' in training_results['training_history']:
            client_participation = {}
            for round_clients in training_results['training_history']['selected_clients']:
                for client_id in round_clients:
                    client_participation[client_id] = client_participation.get(client_id, 0) + 1
            
            for client_id, count in client_participation.items():
                f.write(f"Client {client_id}: {count} rounds\n")
        
        # 评估结果
        f.write("\nEvaluation Results:\n")
        f.write("-" * 30 + "\n")
        if 'average' in eval_results:
            avg_metrics = eval_results['average']
            for metric, value in avg_metrics.items():
                if not metric.endswith('_std'):
                    std_metric = f"{metric}_std"
                    std_value = avg_metrics.get(std_metric, 0.0)
                    f.write(f"{metric.upper()}: {value:.4f} ± {std_value:.4f}\n")
        
        # 各客户端详细结果
        f.write("\nPer-Client Results:\n")
        f.write("-" * 30 + "\n")
        for client_id, result in eval_results.items():
            if client_id == 'average':
                continue
            f.write(f"\nClient {client_id}:\n")
            for metric, value in result.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric.upper()}: {value:.4f}\n")
    
    logger.info(f"Training summary saved to {save_path}")


def plot_confusion_matrices(eval_results: Dict, save_path: str = "results/confusion_matrices.png") -> None:
    """
    绘制混淆矩阵
    
    Args:
        eval_results: 评估结果字典
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    clients = [client_id for client_id in eval_results.keys() if client_id != 'average']
    n_clients = len(clients)
    
    if n_clients == 0:
        return
    
    # 计算子图布局
    cols = min(2, n_clients)
    rows = (n_clients + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_clients == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Confusion Matrices by Client', fontsize=16)
    
    for i, client_id in enumerate(clients):
        row = i // cols
        col = i % cols
        
        if 'confusion_matrix' in eval_results[client_id]:
            cm = eval_results[client_id]['confusion_matrix']
            
            # 绘制混淆矩阵
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Abnormal'],
                       yticklabels=['Normal', 'Abnormal'],
                       ax=axes[row, col])
            axes[row, col].set_title(f'Client {client_id}')
            axes[row, col].set_xlabel('Predicted')
            axes[row, col].set_ylabel('Actual')
    
    # 隐藏多余的子图
    for i in range(n_clients, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrices saved to {save_path}")


if __name__ == "__main__":
    # 测试可视化功能
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    training_history = {
        'rounds': list(range(1, 11)),
        'global_losses': np.random.exponential(1, 10),
        'prototype_distances': np.random.exponential(0.5, 10),
        'eval_aucs': np.random.uniform(0.7, 0.9, 10),
        'selected_clients': [['id_00', 'id_02', 'id_04', 'id_06']] * 10
    }
    
    eval_results = {
        'id_00': {'auc': 0.85, 'f1': 0.80, 'precision': 0.82, 'recall': 0.78},
        'id_02': {'auc': 0.87, 'f1': 0.83, 'precision': 0.85, 'recall': 0.81},
        'id_04': {'auc': 0.84, 'f1': 0.79, 'precision': 0.81, 'recall': 0.77},
        'id_06': {'auc': 0.86, 'f1': 0.82, 'precision': 0.84, 'recall': 0.80},
        'average': {'auc': 0.855, 'auc_std': 0.012, 'f1': 0.81, 'f1_std': 0.017}
    }
    
    # 测试绘图功能
    plot_training_curves(training_history)
    plot_client_metrics(eval_results)
    
    print("Visualization tests completed!")
