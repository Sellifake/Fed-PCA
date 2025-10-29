"""
Fed-ProFiLA-AD Loss Functions and Utilities
实现Fed-ProFiLA-AD的损失函数和工具函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def compute_task_loss(z_batch: torch.Tensor, mu_local: torch.Tensor) -> torch.Tensor:
    """
    计算任务损失 (紧凑性损失)
    将Normal样本的嵌入拉向本地原型
    添加防坍塌机制：方差正则化和范数约束
    
    Args:
        z_batch: 批次特征嵌入 [batch_size, feature_dim]
        mu_local: 本地原型 [feature_dim]
        
    Returns:
        torch.Tensor: 任务损失标量
    """
    # 1. 紧凑性损失：特征到原型的距离
    distances = torch.sum((z_batch - mu_local.unsqueeze(0)) ** 2, dim=1)
    compactness_loss = torch.mean(distances)

    # 2. 方差正则化：防止所有特征坍塌到同一点
    # 计算每一维的方差，要求整体方差不低于阈值
    feature_var_per_dim = torch.var(z_batch, dim=0, unbiased=False)
    min_var = 0.05  # 提高方差下限，鼓励表征动量
    var_shortfall = torch.relu(min_var - feature_var_per_dim)
    var_penalty = var_shortfall.mean() * 50.0  # 更强的权重以显著抑制坍塌

    # 3. 特征范数正则化：鼓励特征模长保持在合理范围
    feature_norms = torch.norm(z_batch, dim=1)
    target_norm = 1.0
    norm_shortfall = torch.relu(target_norm - feature_norms)
    norm_penalty = norm_shortfall.mean() * 10.0

    # 总任务损失 = 紧凑性 + 防坍塌项 + 范数约束
    task_loss = compactness_loss + var_penalty + norm_penalty
    
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
    # 计算本地原型和全局原型之间的L2距离
    diff = mu_local - mu_global
    proto_loss = torch.sum(diff ** 2)
    
    # 添加小的epsilon避免数值不稳定
    epsilon = 1e-8
    proto_loss = proto_loss + epsilon
    
    return proto_loss


def compute_total_loss(
    z_batch: torch.Tensor,
    mu_local: torch.Tensor,
    mu_global: torch.Tensor,
    lambda_proto: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算总损失
    
    Args:
        z_batch: 批次特征嵌入 [batch_size, feature_dim]
        mu_local: 本地原型 [feature_dim]
        mu_global: 全局原型 [feature_dim]
        lambda_proto: 原型对齐损失权重
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (总损失, 任务损失, 原型对齐损失)
    """
    # 计算任务损失
    task_loss = compute_task_loss(z_batch, mu_local)
    
    # 计算原型对齐损失
    proto_loss = compute_prototype_alignment_loss(mu_local, mu_global)
    
    # 计算总损失
    total_loss = task_loss + lambda_proto * proto_loss
    
    return total_loss, task_loss, proto_loss


def compute_local_prototype(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    global_prototype: torch.Tensor,
    device: torch.device,
    device_type_id: int = 0
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
    
    with torch.no_grad():
        for batch_data in dataloader:
            # 解包数据（可能是2个或3个值）
            if len(batch_data) == 3:
                batch_x, batch_y, _ = batch_data  # 忽略device_type
            else:
                batch_x, batch_y = batch_data
            batch_x = batch_x.to(device)
            
            # 前向传播（使用设备类型ID而不是global_prototype）
            device_type_id_tensor = torch.tensor([device_type_id] * batch_x.size(0), 
                                                 dtype=torch.long, device=device)
            features = model(batch_x, device_type_id_tensor)
            all_features.append(features)
    
    # 计算平均特征作为本地原型
    all_features = torch.cat(all_features, dim=0)
    local_prototype = torch.mean(all_features, dim=0)
    
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
    
    # 计算到本地原型的L2距离
    distances = torch.sum((z_test - mu_local) ** 2, dim=1)
    
    # 使用平方根使分数分布更合理
    distances = torch.sqrt(distances + 1e-8)
    
    return distances


class FedProFiLALoss(nn.Module):
    """
    Fed-ProFiLA-AD损失函数模块
    """
    
    def __init__(self, lambda_proto: float = 0.1):
        """
        初始化损失函数
        
        Args:
            lambda_proto: 原型对齐损失权重
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
    client_weights = [w / total_weight for w in client_weights]
    
    # 获取所有参数键
    param_keys = model_states[0].keys()
    
    # 初始化聚合参数
    aggregated_state = {}
    
    for key in param_keys:
        # 计算加权平均
        weighted_sum = None
        for i, state in enumerate(model_states):
            if weighted_sum is None:
                weighted_sum = client_weights[i] * state[key]
            else:
                weighted_sum += client_weights[i] * state[key]
        
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
    client_weights = [w / total_weight for w in client_weights]
    
    # 计算加权平均
    weighted_sum = None
    for i, prototype in enumerate(prototypes):
        if weighted_sum is None:
            weighted_sum = client_weights[i] * prototype
        else:
            weighted_sum += client_weights[i] * prototype
    
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


if __name__ == "__main__":
    # 测试损失函数和工具函数
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试参数
    batch_size = 4
    feature_dim = 128
    
    # 创建测试数据
    z_batch = torch.randn(batch_size, feature_dim)
    mu_local = torch.randn(feature_dim)
    mu_global = torch.randn(feature_dim)
    lambda_proto = 0.1
    
    # 测试任务损失
    print("Testing task loss...")
    task_loss = compute_task_loss(z_batch, mu_local)
    print(f"Task loss: {task_loss.item():.4f}")
    
    # 测试原型对齐损失
    print("\nTesting prototype alignment loss...")
    proto_loss = compute_prototype_alignment_loss(mu_local, mu_global)
    print(f"Prototype alignment loss: {proto_loss.item():.4f}")
    
    # 测试总损失
    print("\nTesting total loss...")
    total_loss, task_loss_comp, proto_loss_comp = compute_total_loss(
        z_batch, mu_local, mu_global, lambda_proto
    )
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Task loss component: {task_loss_comp.item():.4f}")
    print(f"Prototype loss component: {proto_loss_comp.item():.4f}")
    
    # 测试损失函数模块
    print("\nTesting FedProFiLALoss module...")
    loss_fn = FedProFiLALoss(lambda_proto=lambda_proto)
    total_loss_module, task_loss_module, proto_loss_module = loss_fn(z_batch, mu_local, mu_global)
    print(f"Module total loss: {total_loss_module.item():.4f}")
    
    # 测试异常分数计算
    print("\nTesting anomaly score computation...")
    anomaly_scores = compute_anomaly_score(z_batch, mu_local)
    print(f"Anomaly scores shape: {anomaly_scores.shape}")
    print(f"Anomaly scores: {anomaly_scores}")
    
    # 测试模型聚合
    print("\nTesting model aggregation...")
    model_states = [
        {"weight": torch.randn(10, 5), "bias": torch.randn(5)},
        {"weight": torch.randn(10, 5), "bias": torch.randn(5)},
        {"weight": torch.randn(10, 5), "bias": torch.randn(5)}
    ]
    client_weights = [0.5, 0.3, 0.2]
    
    aggregated_state = aggregate_models(model_states, client_weights)
    print(f"Aggregated state keys: {list(aggregated_state.keys())}")
    print(f"Aggregated weight shape: {aggregated_state['weight'].shape}")
    
    # 测试原型聚合
    print("\nTesting prototype aggregation...")
    prototypes = [torch.randn(feature_dim) for _ in range(3)]
    client_weights = [0.5, 0.3, 0.2]
    
    aggregated_prototype = aggregate_prototypes(prototypes, client_weights)
    print(f"Aggregated prototype shape: {aggregated_prototype.shape}")
    
    # 测试全局原型初始化
    print("\nTesting global prototype initialization...")
    global_prototype = initialize_global_prototype(feature_dim, torch.device('cpu'))
    print(f"Initial global prototype shape: {global_prototype.shape}")
    print(f"Initial global prototype sum: {global_prototype.sum().item()}")
    
    # 测试原型维度验证
    print("\nTesting prototype dimension validation...")
    valid_prototypes = [torch.randn(feature_dim) for _ in range(3)]
    invalid_prototypes = [torch.randn(feature_dim), torch.randn(feature_dim + 1), torch.randn(feature_dim)]
    
    is_valid = validate_prototype_dimensions(valid_prototypes, feature_dim)
    print(f"Valid prototypes validation: {is_valid}")
    
    is_invalid = validate_prototype_dimensions(invalid_prototypes, feature_dim)
    print(f"Invalid prototypes validation: {is_invalid}")
    
    print("\nAll tests passed!")
