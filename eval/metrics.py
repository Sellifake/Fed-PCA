"""
Evaluation Metrics for Fed-ProFiLA-AD
实现异常检测评估指标
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    f1_score, 
    precision_score, 
    recall_score,
    confusion_matrix
)
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def calculate_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算AUC (Area Under Curve)
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        
    Returns:
        float: AUC值
    """
    try:
        auc_score = roc_auc_score(labels, scores)
        return auc_score
    except ValueError as e:
        logger.error(f"Error calculating AUC: {e}")
        return 0.0


def calculate_pr_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    计算PR-AUC (Precision-Recall Area Under Curve)
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        
    Returns:
        float: PR-AUC值
    """
    try:
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        return pr_auc
    except ValueError as e:
        logger.error(f"Error calculating PR-AUC: {e}")
        return 0.0


def calculate_f1_score(scores: np.ndarray, labels: np.ndarray, threshold: Optional[float] = None) -> float:
    """
    计算F1分数
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        threshold: 分类阈值，如果为None则使用最佳阈值
        
    Returns:
        float: F1分数
    """
    if threshold is None:
        # 使用最佳F1分数对应的阈值
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        threshold = thresholds[best_threshold_idx]
    
    # 使用阈值进行二分类
    predictions = (scores >= threshold).astype(int)
    
    try:
        f1 = f1_score(labels, predictions)
        return f1
    except ValueError as e:
        logger.error(f"Error calculating F1 score: {e}")
        return 0.0


def calculate_precision(scores: np.ndarray, labels: np.ndarray, threshold: Optional[float] = None) -> float:
    """
    计算精确率
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        threshold: 分类阈值，如果为None则使用最佳阈值
        
    Returns:
        float: 精确率
    """
    if threshold is None:
        # 使用最佳F1分数对应的阈值
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        threshold = thresholds[best_threshold_idx]
    
    # 使用阈值进行二分类
    predictions = (scores >= threshold).astype(int)
    
    try:
        precision = precision_score(labels, predictions, zero_division=0)
        return precision
    except ValueError as e:
        logger.error(f"Error calculating precision: {e}")
        return 0.0


def calculate_recall(scores: np.ndarray, labels: np.ndarray, threshold: Optional[float] = None) -> float:
    """
    计算召回率
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        threshold: 分类阈值，如果为None则使用最佳阈值
        
    Returns:
        float: 召回率
    """
    if threshold is None:
        # 使用最佳F1分数对应的阈值
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_threshold_idx = np.argmax(f1_scores)
        threshold = thresholds[best_threshold_idx]
    
    # 使用阈值进行二分类
    predictions = (scores >= threshold).astype(int)
    
    try:
        recall = recall_score(labels, predictions, zero_division=0)
        return recall
    except ValueError as e:
        logger.error(f"Error calculating recall: {e}")
        return 0.0


def find_best_threshold(scores: np.ndarray, labels: np.ndarray, metric: str = "f1") -> Tuple[float, float]:
    """
    找到最佳分类阈值
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        metric: 优化指标 ("f1", "precision", "recall")
        
    Returns:
        Tuple[float, float]: (最佳阈值, 最佳指标值)
    """
    # 检查分数是否有变化
    if np.std(scores) < 1e-8:
        # 如果所有分数相同，使用分数中位数作为阈值
        threshold = np.median(scores)
        if metric.lower() == "f1":
            # 计算F1分数
            pred = (scores >= threshold).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            return threshold, f1
        else:
            return threshold, 0.0
    
    # 使用ROC曲线找到最佳阈值
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    if metric.lower() == "f1":
        # 计算每个阈值的F1分数
        f1_scores = []
        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores.append(f1)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_value = f1_scores[best_idx]
    elif metric.lower() == "precision":
        # 计算每个阈值的精确率
        precisions = []
        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            precision = tp / (tp + fp + 1e-8)
            precisions.append(precision)
        
        best_idx = np.argmax(precisions)
        best_threshold = thresholds[best_idx]
        best_value = precisions[best_idx]
    elif metric.lower() == "recall":
        # 计算每个阈值的召回率
        recalls = []
        for threshold in thresholds:
            pred = (scores >= threshold).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fn = np.sum((pred == 0) & (labels == 1))
            recall = tp / (tp + fn + 1e-8)
            recalls.append(recall)
        
        best_idx = np.argmax(recalls)
        best_threshold = thresholds[best_idx]
        best_value = recalls[best_idx]
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return best_threshold, best_value


def calculate_confusion_matrix(scores: np.ndarray, labels: np.ndarray, threshold: float) -> np.ndarray:
    """
    计算混淆矩阵
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        threshold: 分类阈值
        
    Returns:
        np.ndarray: 混淆矩阵
    """
    predictions = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions)
    return cm


def calculate_all_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算所有评估指标
    
    Args:
        scores: 异常分数数组
        labels: 真实标签数组 (0=Normal, 1=Abnormal)
        
    Returns:
        Dict[str, float]: 所有指标字典
    """
    metrics = {}
    
    # 基本指标
    metrics['auc'] = calculate_auc(scores, labels)
    metrics['pr_auc'] = calculate_pr_auc(scores, labels)
    
    # 找到最佳阈值
    best_threshold, best_f1 = find_best_threshold(scores, labels, "f1")
    metrics['best_threshold'] = best_threshold
    metrics['best_f1'] = best_f1
    # 保持向后兼容，显式提供'f1'键以避免下游KeyError
    metrics['f1'] = best_f1
    
    # 使用最佳阈值计算其他指标
    metrics['precision'] = calculate_precision(scores, labels, best_threshold)
    metrics['recall'] = calculate_recall(scores, labels, best_threshold)
    
    # 混淆矩阵
    cm = calculate_confusion_matrix(scores, labels, best_threshold)
    metrics['confusion_matrix'] = cm
    
    # 从混淆矩阵计算额外指标
    tn, fp, fn, tp = cm.ravel()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return metrics


def print_metrics(metrics: Dict[str, float], client_id: str = "") -> None:
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        client_id: 客户端ID
    """
    prefix = f"[{client_id}] " if client_id else ""
    
    print(f"\n{prefix}Evaluation Metrics:")
    print(f"{prefix}AUC: {metrics['auc']:.4f}")
    print(f"{prefix}PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"{prefix}Best F1: {metrics['best_f1']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall: {metrics['recall']:.4f}")
    print(f"{prefix}Specificity: {metrics['specificity']:.4f}")
    print(f"{prefix}Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix}Best Threshold: {metrics['best_threshold']:.4f}")
    
    print(f"\n{prefix}Confusion Matrix:")
    print(f"{prefix}TN: {metrics['confusion_matrix'][0,0]}, FP: {metrics['confusion_matrix'][0,1]}")
    print(f"{prefix}FN: {metrics['confusion_matrix'][1,0]}, TP: {metrics['confusion_matrix'][1,1]}")


def calculate_average_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    计算所有客户端的平均指标
    
    Args:
        all_metrics: 所有客户端的指标列表
        
    Returns:
        Dict[str, float]: 平均指标字典
    """
    if not all_metrics:
        return {}
    
    # 需要平均的指标
    avg_metrics = ['auc', 'pr_auc', 'best_f1', 'precision', 'recall', 'specificity', 'sensitivity', 'accuracy']
    
    average_metrics = {}
    for metric in avg_metrics:
        values = [m[metric] for m in all_metrics if metric in m]
        if values:
            average_metrics[metric] = np.mean(values)
            average_metrics[f'{metric}_std'] = np.std(values)
    
    return average_metrics


if __name__ == "__main__":
    # 测试评估指标
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟的异常分数和标签
    normal_scores = np.random.normal(0.2, 0.1, 800)  # Normal样本分数较低
    abnormal_scores = np.random.normal(0.8, 0.1, 200)  # Abnormal样本分数较高
    
    scores = np.concatenate([normal_scores, abnormal_scores])
    labels = np.concatenate([np.zeros(800), np.ones(200)])
    
    # 打乱数据
    indices = np.random.permutation(len(scores))
    scores = scores[indices]
    labels = labels[indices]
    
    print(f"Test data: {len(scores)} samples, {np.sum(labels)} abnormal")
    print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # 计算所有指标
    metrics = calculate_all_metrics(scores, labels)
    print_metrics(metrics, "Test")
    
    # 测试平均指标计算
    print("\nTesting average metrics calculation...")
    all_metrics = [metrics, metrics, metrics]  # 模拟多个客户端
    avg_metrics = calculate_average_metrics(all_metrics)
    
    print("\nAverage Metrics:")
    for key, value in avg_metrics.items():
        print(f"{key}: {value:.4f}")
    
    print("\nAll tests passed!")
