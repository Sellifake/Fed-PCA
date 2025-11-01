"""
Inference Module for Fed-ProFiLA-AD
实现异常检测推理和评估
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from methods.fed_profila_ad import compute_anomaly_score
from eval.metrics import calculate_all_metrics, print_metrics, calculate_average_metrics

logger = logging.getLogger(__name__)


def evaluate_client(
    model: nn.Module,
    test_loader: DataLoader,
    global_prototype: torch.Tensor,
    local_prototype: torch.Tensor,
    device: torch.device,
    client_id: str = ""
) -> Dict[str, float]:
    """
    评估单个客户端
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        global_prototype: 全局原型
        local_prototype: 本地原型
        device: 设备
        client_id: 客户端ID
        
    Returns:
        Dict[str, float]: 评估指标字典
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            global_prototype = global_prototype.to(device)
            local_prototype = local_prototype.to(device)
            
            # 前向传播获取特征嵌入（使用全局原型）
            z_test = model(batch_x, global_prototype)
            
            # 计算异常分数
            scores = compute_anomaly_score(z_test, local_prototype)
            
            # 收集结果
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy() if isinstance(batch_y, torch.Tensor) else batch_y)
    
    # 转换为numpy数组
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    metrics = calculate_all_metrics(all_scores, all_labels)
    
    # 不打印详细结果（简化输出）
    
    return metrics


def evaluate_all_clients(
    models: Dict[str, nn.Module],
    test_loaders: Dict[str, DataLoader],
    global_prototype: torch.Tensor,
    local_prototypes: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    评估所有客户端
    
    Args:
        models: 客户端模型字典
        test_loaders: 客户端测试数据加载器字典
        global_prototype: 全局原型
        local_prototypes: 客户端本地原型字典
        device: 设备
        
    Returns:
        Dict[str, Dict[str, float]]: 所有客户端的评估指标字典
    """
    all_metrics = {}
    
    logger.info("Starting evaluation of all clients...")
    
    for client_id in models.keys():
        logger.info(f"Evaluating client {client_id}...")
        
        # 评估单个客户端
        metrics = evaluate_client(
            model=models[client_id],
            test_loader=test_loaders[client_id],
            global_prototype=global_prototype,
            local_prototype=local_prototypes[client_id],
            device=device,
            client_id=client_id
        )
        
        all_metrics[client_id] = metrics
    
    # 计算平均指标
    logger.info("Calculating average metrics...")
    metrics_list = list(all_metrics.values())
    avg_metrics = calculate_average_metrics(metrics_list)
    
    # 打印平均指标
    print("\n" + "="*50)
    print("AVERAGE METRICS ACROSS ALL CLIENTS")
    print("="*50)
    for key, value in avg_metrics.items():
        if not key.endswith('_std'):
            std_key = f"{key}_std"
            std_value = avg_metrics.get(std_key, 0.0)
            print(f"{key}: {value:.4f} ± {std_value:.4f}")
    
    all_metrics['average'] = avg_metrics
    
    return all_metrics


def compute_anomaly_scores_batch(
    model: nn.Module,
    batch_x: torch.Tensor,
    global_prototype: torch.Tensor,
    local_prototype: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    批量计算异常分数
    
    Args:
        model: 模型
        batch_x: 输入批次
        global_prototype: 全局原型
        local_prototype: 本地原型
        device: 设备
        
    Returns:
        torch.Tensor: 异常分数
    """
    model.eval()
    
    with torch.no_grad():
        batch_x = batch_x.to(device)
        global_prototype = global_prototype.to(device)
        local_prototype = local_prototype.to(device)
        
        # 前向传播（使用全局原型）
        z_test = model(batch_x, global_prototype)
        
        # 计算异常分数
        scores = compute_anomaly_score(z_test, local_prototype)
        
    return scores


def predict_anomaly(
    model: nn.Module,
    x: torch.Tensor,
    global_prototype: torch.Tensor,
    local_prototype: torch.Tensor,
    threshold: float,
    device: torch.device
) -> Tuple[bool, float]:
    """
    预测单个样本是否为异常
    
    Args:
        model: 模型
        x: 输入样本
        global_prototype: 全局原型
        local_prototype: 本地原型
        threshold: 异常阈值
        device: 设备
        
    Returns:
        Tuple[bool, float]: (是否为异常, 异常分数)
    """
    # 添加batch维度
    if x.dim() == 3:  # [channels, height, width]
        x = x.unsqueeze(0)  # [1, channels, height, width]
    
    # 计算异常分数
    score = compute_anomaly_scores_batch(
        model, x, global_prototype, local_prototype, device
    )
    
    # 判断是否为异常
    is_anomaly = score.item() >= threshold
    
    return is_anomaly, score.item()


def evaluate_with_different_thresholds(
    model: nn.Module,
    test_loader: DataLoader,
    global_prototype: torch.Tensor,
    local_prototype: torch.Tensor,
    device: torch.device,
    thresholds: List[float]
) -> Dict[float, Dict[str, float]]:
    """
    使用不同阈值评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        global_prototype: 全局原型
        local_prototype: 本地原型
        device: 设备
        thresholds: 阈值列表
        
    Returns:
        Dict[float, Dict[str, float]]: 每个阈值对应的指标字典
    """
    model.eval()
    all_scores = []
    all_labels = []
    
    # 收集所有分数和标签
    with torch.no_grad():
        for batch_data in test_loader:
            # 解包数据（可能是2个或3个值）
            if len(batch_data) == 3:
                batch_x, batch_y, _ = batch_data  # 忽略device_type
            else:
                batch_x, batch_y = batch_data
            batch_x = batch_x.to(device)
            global_prototype = global_prototype.to(device)
            local_prototype = local_prototype.to(device)
            
            # 前向传播（使用全局原型）
            z_test = model(batch_x, global_prototype)
            scores = compute_anomaly_score(z_test, local_prototype)
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # 对每个阈值计算指标
    threshold_metrics = {}
    
    for threshold in thresholds:
        # 使用阈值进行二分类
        predictions = (all_scores >= threshold).astype(int)
        
        # 计算基本指标
        from eval.metrics import calculate_all_metrics
        metrics = calculate_all_metrics(all_scores, all_labels)
        
        # 使用指定阈值重新计算
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        precision = precision_score(all_labels, predictions, zero_division=0)
        recall = recall_score(all_labels, predictions, zero_division=0)
        f1 = f1_score(all_labels, predictions, zero_division=0)
        accuracy = accuracy_score(all_labels, predictions)
        
        threshold_metrics[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'threshold': threshold
        }
    
    return threshold_metrics


def save_evaluation_results(
    results: Dict[str, Dict[str, float]],
    save_path: str
) -> None:
    """
    保存评估结果
    
    Args:
        results: 评估结果字典
        save_path: 保存路径
    """
    import json
    import os
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换numpy数组为列表以便JSON序列化
    serializable_results = {}
    for client_id, metrics in results.items():
        serializable_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable_metrics[key] = value.tolist()
            else:
                serializable_metrics[key] = value
        serializable_results[client_id] = serializable_metrics
    
    # 保存为JSON文件
    with open(save_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {save_path}")


def load_evaluation_results(load_path: str) -> Dict[str, Dict[str, float]]:
    """
    加载评估结果
    
    Args:
        load_path: 加载路径
        
    Returns:
        Dict[str, Dict[str, float]]: 评估结果字典
    """
    import json
    
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Evaluation results loaded from {load_path}")
    return results


if __name__ == "__main__":
    # 测试推理模块
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    device = torch.device('cpu')
    batch_size = 4
    n_mels = 128
    time_frames = 32
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_mels * time_frames, 128)
        
        def forward(self, x, global_prototype):
            x = x.view(x.size(0), -1)
            return self.linear(x)
    
    model = MockModel()
    
    # 创建模拟数据
    x = torch.randn(batch_size, 1, n_mels, time_frames)
    global_prototype = torch.randn(128)
    local_prototype = torch.randn(128)
    
    # 测试异常分数计算
    print("Testing anomaly score computation...")
    scores = compute_anomaly_scores_batch(
        model, x, global_prototype, local_prototype, device
    )
    print(f"Anomaly scores shape: {scores.shape}")
    print(f"Anomaly scores: {scores}")
    
    # 测试异常预测
    print("\nTesting anomaly prediction...")
    threshold = 0.5
    is_anomaly, score = predict_anomaly(
        model, x[0], global_prototype, local_prototype, threshold, device
    )
    print(f"Is anomaly: {is_anomaly}, Score: {score:.4f}")
    
    print("\nAll tests passed!")
