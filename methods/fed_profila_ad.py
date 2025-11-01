"""
Fed-ProFiLA-AD Loss Functions and Utilities (基础版本)
实现Fed-ProFiLA-AD的损失函数和工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def _l2_normalize_rows(x: torch.Tensor) -> torch.Tensor:
    """
    对最后一维做L2归一化。
    """
    return F.normalize(x, p=2, dim=-1)


def compute_task_loss(z_batch: torch.Tensor, mu_local: torch.Tensor) -> torch.Tensor:
    """
    计算任务损失 (紧凑性损失)
    将Normal样本的嵌入拉向本地原型
    
    Args:
        z_batch: 批次特征嵌入 [batch_size, feature_dim]
        mu_local: 本地原型 [feature_dim]
        
    Returns:
        torch.Tensor: 任务损失标量
    """
    # 归一化特征与原型，提升稳定性与可分性
    z_batch = _l2_normalize_rows(z_batch)
    mu_local = _l2_normalize_rows(mu_local.unsqueeze(0)).squeeze(0)

    # 紧凑性损失：特征到原型的L2距离平方
    diff = z_batch - mu_local.unsqueeze(0)  # [batch_size, feature_dim]
    distances = torch.sum(diff ** 2, dim=1)  # [batch_size]
    task_loss = torch.mean(distances)
    
    return task_loss


def compute_prototype_alignment_loss(mu_local: torch.Tensor, mu_global: torch.Tensor) -> torch.Tensor:
    """
    计算原型对齐损失
    将本地原型拉向全局原型
    
    Args:
        mu_local: 本地原型 [feature_dim]
        mu_global: 全局原型 [feature_dim]
        
    Returns:
        torch.Tensor: 原型对齐损失标量
    """
    # 归一化原型后计算距离
    mu_local = _l2_normalize_rows(mu_local)
    mu_global = _l2_normalize_rows(mu_global)

    diff = mu_local - mu_global
    proto_loss = torch.sum(diff ** 2)
    
    return proto_loss


def compute_contrastive_loss(
    z_batch: torch.Tensor,
    y_batch: torch.Tensor,
    mu_local: torch.Tensor,
    margin: float = 0.8
) -> torch.Tensor:
    """
    对比损失：
    - 正样本(正常=0)：拉近到本地原型（与task一致，这里不重复计入，由task_loss负责）
    - 负样本(异常=1)：推远离本地原型，采用hinge式 margin: relu(m - d)^2
    """
    if y_batch is None:
        return torch.tensor(0.0, device=z_batch.device)
    # 归一化
    z_batch = _l2_normalize_rows(z_batch)
    mu_local = _l2_normalize_rows(mu_local)

    # 距离
    diff = z_batch - mu_local
    dists = torch.sqrt(torch.sum(diff ** 2, dim=1) + 1e-8)  # [B]

    # 仅对异常样本计算push-away
    mask_abn = (y_batch > 0.5).float()
    if mask_abn.sum() == 0:
        return torch.tensor(0.0, device=z_batch.device)

    hinge = torch.relu(margin - dists)  # d < m 才有惩罚
    loss_abn = (hinge ** 2) * mask_abn
    return loss_abn.sum() / (mask_abn.sum() + 1e-8)


def compute_batch_separation_loss(
    z_batch: torch.Tensor,
    y_batch: Optional[torch.Tensor],
    margin: float = 0.8
) -> torch.Tensor:
    """
    批内“正常中心 vs 异常样本”分离损失：
    - 计算正常样本的批内中心 c_n
    - 对每个异常样本，施加 hinge: relu(margin - ||z_abn - c_n||)^2
    """
    if y_batch is None:
        return torch.tensor(0.0, device=z_batch.device)
    # 归一化
    z_batch = _l2_normalize_rows(z_batch)
    y = (y_batch > 0.5).float()
    mask_norm = (y == 0)
    mask_abn = (y == 1)
    if mask_norm.sum() == 0 or mask_abn.sum() == 0:
        return torch.tensor(0.0, device=z_batch.device)
    c_n = z_batch[mask_norm].mean(dim=0)
    # 归一化中心
    c_n = _l2_normalize_rows(c_n.unsqueeze(0)).squeeze(0)
    z_abn = z_batch[mask_abn]
    dists = torch.sqrt(torch.sum((z_abn - c_n.unsqueeze(0)) ** 2, dim=1) + 1e-8)
    hinge = torch.relu(margin - dists)
    return torch.mean(hinge ** 2)


def compute_supervised_contrastive_loss(
    z_batch: torch.Tensor,
    y_batch: Optional[torch.Tensor],
    temperature: float = 0.2
) -> torch.Tensor:
    """
    监督式对比损失（SupCon）：同类为正样本、异类为负样本。
    要求batch中同时存在至少两个类别，否则返回0。
    参考: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
    """
    if y_batch is None:
        return torch.tensor(0.0, device=z_batch.device)
    y = y_batch.view(-1)
    if y.dim() != 1 or z_batch.size(0) < 2:
        return torch.tensor(0.0, device=z_batch.device)
    # 归一化
    z = _l2_normalize_rows(z_batch)
    # 相似度矩阵
    sim = torch.matmul(z, z.t()) / max(1e-6, temperature)  # [B,B]
    # 掩码
    labels = (y > 0.5).long()
    mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float().to(z.device)  # 同类为1
    # 去除自身
    logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=z.device)
    mask = mask * logits_mask
    # 归一化softmax分母
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    # 只对正样本对求平均
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    loss = -mean_log_prob_pos.mean()
    if not torch.isfinite(loss):
        return torch.tensor(0.0, device=z.device)
    return loss


def compute_total_loss(
    z_batch: torch.Tensor,
    mu_local: torch.Tensor,
    mu_global: torch.Tensor,
    lambda_proto: float = 0.01,
    # 可选对比学习项
    y_batch: Optional[torch.Tensor] = None,
    lambda_contrastive: float = 0.5,
    contrastive_margin: float = 0.8,
    # 批内分离项
    lambda_separation: float = 0.5,
    separation_margin: float = 0.8,
    # 监督式对比
    lambda_supcon: float = 1.0,
    temperature: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算总损失
    
    Args:
        z_batch: 批次特征嵌入 [batch_size, feature_dim]
        mu_local: 本地原型 [feature_dim]
        mu_global: 全局原型 [feature_dim]
        lambda_proto: 原型对齐损失权重（默认0.01，较小值更稳定）
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (总损失, 任务损失, 原型对齐损失)
    """
    # 计算任务损失
    task_loss = compute_task_loss(z_batch, mu_local)
    
    # 计算原型对齐损失（只在训练时计算，不阻止梯度）
    proto_loss = compute_prototype_alignment_loss(mu_local, mu_global)

    # 计算对比损失（推远异常）
    contrastive_loss = compute_contrastive_loss(z_batch, y_batch, mu_local, margin=contrastive_margin)
    # 计算批内分离损失（异常远离正常中心）
    separation_loss = compute_batch_separation_loss(z_batch, y_batch, margin=separation_margin)

    # SupCon损失
    supcon_loss = compute_supervised_contrastive_loss(z_batch, y_batch, temperature=temperature)

    # 计算总损失
    total_loss = (
        task_loss
        + lambda_proto * proto_loss
        + lambda_contrastive * contrastive_loss
        + lambda_separation * separation_loss
        + lambda_supcon * supcon_loss
    )

    return total_loss, task_loss, proto_loss, contrastive_loss, separation_loss, supcon_loss


def compute_local_prototype(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    global_prototype: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    计算本地原型
    通过所有本地Normal数据的前向传播计算平均特征
    
    Args:
        model: 模型
        dataloader: 数据加载器
        global_prototype: 全局原型
        device: 设备
        
    Returns:
        torch.Tensor: 本地原型 [feature_dim]
    """
    model.eval()
    all_features = []
    normal_count = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # 仅使用Normal样本(y==0)计算本地原型，避免异常样本污染
            if isinstance(batch_y, torch.Tensor):
                mask = (batch_y == 0)
            else:
                # batch_y 可能是list/np，转为Tensor处理
                mask = torch.tensor(batch_y, device='cpu') == 0
            if mask.sum().item() == 0:
                continue

            batch_x = batch_x[mask].to(device)
            global_prototype_batch = global_prototype.to(device)

            features = model(batch_x, global_prototype_batch)
            all_features.append(features)
            normal_count += batch_x.size(0)
    
    # 计算平均特征作为本地原型（对每个样本特征先归一化，再求均值，最后再归一化）
    if len(all_features) == 0:
        logger.warning("No normal features collected; falling back to using all samples if available")
        # 回退：若训练批次无正常样本，则尝试用全部样本估计
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(device)
                global_prototype_batch = global_prototype.to(device)
                features = model(batch_x, global_prototype_batch)
                all_features.append(features)
        if len(all_features) == 0:
            logger.warning("No features collected at all, returning zero prototype")
            return torch.zeros_like(global_prototype).to(device)
    
    all_features = torch.cat(all_features, dim=0)
    all_features = _l2_normalize_rows(all_features)
    local_prototype = torch.mean(all_features, dim=0)
    local_prototype = _l2_normalize_rows(local_prototype)

    logger.debug(f"Computed local prototype with {normal_count} normal samples, dim={local_prototype.shape[0]}")
    
    return local_prototype


def compute_anomaly_score(
    z_test: torch.Tensor,
    mu_local: torch.Tensor
) -> torch.Tensor:
    """
    计算异常分数
    使用到本地原型的L2距离作为异常分数
    
    Args:
        z_test: 测试特征嵌入 [batch_size, feature_dim]
        mu_local: 本地原型 [feature_dim]
        
    Returns:
        torch.Tensor: 异常分数 [batch_size]
    """
    # 确保原型有正确的维度
    if mu_local.dim() == 1:
        mu_local = mu_local.unsqueeze(0)  # [1, feature_dim]
    
    # 归一化后计算距离
    z_test = _l2_normalize_rows(z_test)
    mu_local = _l2_normalize_rows(mu_local)

    diff = z_test - mu_local  # [batch_size, feature_dim]
    distances = torch.sum(diff ** 2, dim=1)  # [batch_size]
    
    # 使用平方根使分数分布更合理
    distances = torch.sqrt(distances + 1e-8)
    
    return distances


class FedProFiLALoss(nn.Module):
    """
    Fed-ProFiLA-AD损失函数模块
    """
    
    def __init__(self, lambda_proto: float = 0.01):
        """
        初始化损失函数
        
        Args:
            lambda_proto: 原型对齐损失权重（默认0.01）
        """
        super(FedProFiLALoss, self).__init__()
        self.lambda_proto = lambda_proto
    
    def forward(
        self,
        z_batch: torch.Tensor,
        mu_local: torch.Tensor,
        mu_global: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z_batch: 批次特征嵌入
            mu_local: 本地原型
            mu_global: 全局原型
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (总损失, 任务损失, 原型对齐损失)
        """
        return compute_total_loss(z_batch, mu_local, mu_global, self.lambda_proto)


def aggregate_models(model_states: list, client_weights: list = None) -> dict:
    """
    聚合模型参数 (FedAvg)
    
    Args:
        model_states: 客户端模型状态字典列表
        client_weights: 客户端权重列表 (可选，默认为均匀权重)
        
    Returns:
        dict: 聚合后的模型状态字典
    """
    if not model_states:
        raise ValueError("No model states provided for aggregation")
    
    # 如果没有提供权重，使用均匀权重
    if client_weights is None:
        client_weights = [1.0 / len(model_states)] * len(model_states)
    
    # 确保权重归一化
    total_weight = sum(client_weights)
    if total_weight == 0:
        client_weights = [1.0 / len(model_states)] * len(model_states)
    else:
        client_weights = [w / total_weight for w in client_weights]
    
    # 获取所有参数键
    param_keys = model_states[0].keys()
    
    # 初始化聚合参数
    aggregated_state = {}
    
    for key in param_keys:
        # 计算加权平均
        weighted_sum = None
        for i, state in enumerate(model_states):
            if key not in state:
                continue
            if weighted_sum is None:
                weighted_sum = client_weights[i] * state[key]
            else:
                weighted_sum += client_weights[i] * state[key]
        
        if weighted_sum is not None:
            aggregated_state[key] = weighted_sum
    
    return aggregated_state


def aggregate_prototypes(prototypes: list, client_weights: list = None) -> torch.Tensor:
    """
    聚合原型 (加权平均)
    
    Args:
        prototypes: 客户端原型列表
        client_weights: 客户端权重列表 (可选，默认为均匀权重)
        
    Returns:
        torch.Tensor: 聚合后的全局原型
    """
    if not prototypes:
        raise ValueError("No prototypes provided for aggregation")
    
    # 如果没有提供权重，使用均匀权重
    if client_weights is None:
        client_weights = [1.0 / len(prototypes)] * len(prototypes)
    
    # 确保权重归一化
    total_weight = sum(client_weights)
    if total_weight == 0:
        client_weights = [1.0 / len(prototypes)] * len(prototypes)
    else:
        client_weights = [w / total_weight for w in client_weights]
    
    # 计算加权平均
    weighted_sum = None
    for i, prototype in enumerate(prototypes):
        if weighted_sum is None:
            weighted_sum = client_weights[i] * prototype
        else:
            weighted_sum += client_weights[i] * prototype
    
    # 归一化聚合后的原型，保证向量尺度稳定
    weighted_sum = _l2_normalize_rows(weighted_sum)
    return weighted_sum


def initialize_global_prototype(feature_dim: int, device: torch.device) -> torch.Tensor:
    """
    初始化全局原型
    
    Args:
        feature_dim: 特征维度
        device: 设备
        
    Returns:
        torch.Tensor: 初始化的全局原型
    """
    # 使用小的随机值初始化，而不是零向量
    # 这样可以提供更好的学习起点
    global_prototype = torch.randn(feature_dim, device=device) * 0.1
    
    return global_prototype


def validate_prototype_dimensions(prototypes: list, expected_dim: int) -> bool:
    """
    验证原型维度
    
    Args:
        prototypes: 原型列表
        expected_dim: 期望的维度
        
    Returns:
        bool: 是否所有原型维度都正确
    """
    for i, prototype in enumerate(prototypes):
        if prototype.shape[0] != expected_dim:
            logger.error(f"Prototype {i} has dimension {prototype.shape[0]}, expected {expected_dim}")
            return False
    
    return True
