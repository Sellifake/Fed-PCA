"""
Server Training Loop for Fed-ProFiLA-AD
实现服务器聚合循环
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import random

from methods.fed_profila_ad import (
    aggregate_models,
    aggregate_prototypes,
    initialize_global_prototype
)

logger = logging.getLogger(__name__)


class Server:
    """
    联邦学习服务器类
    实现Fed-ProFiLA-AD的服务器聚合逻辑
    """
    
    SHARED_PREFIXES = ("film_generator", "encoder")

    def __init__(
        self,
        global_model: nn.Module,
        device: torch.device,
        client_selection_strategy: str = "all",
        client_selection_fraction: float = 1.0,
        feature_dim: int = 128
    ):
        """
        初始化服务器
        
        Args:
            global_model: 全局模型
            device: 设备
            client_selection_strategy: 客户端选择策略
            client_selection_fraction: 客户端选择比例
            feature_dim: 特征维度
        """
        self.global_model = global_model.to(device)
        self.device = device
        self.client_selection_strategy = client_selection_strategy
        self.client_selection_fraction = client_selection_fraction
        self.feature_dim = feature_dim
        
        # 初始化全局原型
        self.global_prototype = initialize_global_prototype(feature_dim, device)
        
        # 训练历史
        self.training_history = {
            'rounds': [],
            'global_losses': [],
            'prototype_distances': [],
            'selected_clients': []
        }
        
        logger.info(f"Server initialized on device {device}")
        logger.info(f"Global prototype initialized with shape {self.global_prototype.shape}")
    
    def _is_shared_parameter(self, name: str) -> bool:
        """判断参数是否属于共享部分"""
        return name.startswith(Server.SHARED_PREFIXES)

    def _extract_shared_state_dict(
        self,
        state_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """提取共享参数的状态字典"""
        if state_dict is None:
            state_dict = self.global_model.state_dict()
        return {
            name: param.clone()
            for name, param in state_dict.items()
            if self._is_shared_parameter(name)
        }

    def select_clients(self, available_clients: List[str]) -> List[str]:
        """
        选择参与训练的客户端
        
        Args:
            available_clients: 可用客户端列表
            
        Returns:
            List[str]: 选中的客户端列表
        """
        if self.client_selection_strategy.lower() == "all":
            selected_clients = available_clients
        elif self.client_selection_strategy.lower() == "random":
            num_clients = max(1, int(len(available_clients) * self.client_selection_fraction))
            selected_clients = random.sample(available_clients, num_clients)
        else:
            raise ValueError(f"Unknown client selection strategy: {self.client_selection_strategy}")
        
        logger.info(f"已选择 {len(selected_clients)} 个客户端: {selected_clients}")
        return selected_clients
    
    def aggregate_models(self, client_models: Dict[str, Dict[str, torch.Tensor]], 
                        client_weights: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        聚合客户端模型参数
        
        Args:
            client_models: 客户端模型参数字典
            client_weights: 客户端权重大典
            
        Returns:
            Dict[str, torch.Tensor]: 聚合后的模型参数
        """
        # 提取模型状态和权重
        model_states = list(client_models.values())
        weights = [client_weights[client_id] for client_id in client_models.keys()]
        
        # 聚合模型参数
        aggregated_state = aggregate_models(model_states, weights)
        
        # 更新全局模型
        filtered_state = {
            name: param
            for name, param in aggregated_state.items()
            if self._is_shared_parameter(name)
        }
        if filtered_state:
            self.global_model.load_state_dict(filtered_state, strict=False)
        
        logger.info("全局模型聚合成功")
        return filtered_state
    
    def aggregate_prototypes(self, client_prototypes: Dict[str, torch.Tensor], 
                           client_weights: Dict[str, float]) -> torch.Tensor:
        """
        聚合客户端原型
        
        Args:
            client_prototypes: 客户端原型字典
            client_weights: 客户端权重大典
            
        Returns:
            torch.Tensor: 聚合后的全局原型
        """
        # 提取原型和权重
        prototypes = list(client_prototypes.values())
        weights = [client_weights[client_id] for client_id in client_prototypes.keys()]
        
        # 聚合原型
        aggregated_prototype = aggregate_prototypes(prototypes, weights)
        
        # 更新全局原型
        self.global_prototype = aggregated_prototype
        
        logger.info(f"全局原型聚合成功，形状: {self.global_prototype.shape}")
        return aggregated_prototype
    
    def get_global_model_state(self) -> Dict[str, torch.Tensor]:
        """
        获取全局模型状态
        
        Returns:
            Dict[str, torch.Tensor]: 全局模型状态
        """
        return self._extract_shared_state_dict()
    
    def get_global_prototype(self) -> torch.Tensor:
        """
        获取全局原型
        
        Returns:
            torch.Tensor: 全局原型
        """
        return self.global_prototype
    
    def run_federation(
        self,
        client_manager,
        num_rounds: int = 50,
        local_epochs: int = 5,
        lambda_proto: float = 0.1,
        eval_frequency: int = 5
    ) -> Dict[str, any]:
        """
        运行联邦学习训练
        
        Args:
            client_manager: 客户端管理器
            num_rounds: 联邦学习轮数
            local_epochs: 本地训练轮数
            lambda_proto: 原型对齐损失权重
            eval_frequency: 评估频率
            
        Returns:
            Dict[str, any]: 训练结果
        """
        logger.info(f"Starting federation training for {num_rounds} rounds...")
        
        # 获取可用客户端
        available_clients = list(client_manager.get_all_clients().keys())
        client_weights = client_manager.get_client_weights()
        
        # 初始化训练历史
        self.training_history = {
            'rounds': [],
            'global_losses': [],
            'prototype_distances': [],
            'selected_clients': []
        }
        
        # 联邦学习主循环
        for round_idx in range(num_rounds):
            logger.info(f"\n{'='*50}")
            logger.info(f"FEDERATION ROUND {round_idx + 1}/{num_rounds}")
            logger.info(f"{'='*50}")
            
            round_start_time = time.time()
            
            # 步骤A: 服务器广播
            logger.info("步骤A: 服务器广播...")
            global_theta = self.get_global_model_state()
            global_mu = self.get_global_prototype()
            
            # 选择客户端
            selected_clients = self.select_clients(available_clients)
            
            # 步骤B: 客户端本地更新
            logger.info("步骤B: 客户端本地更新...")
            client_results, client_prototypes = client_manager.train_clients(
                global_theta=global_theta,
                global_mu=global_mu,
                selected_clients=selected_clients,
                local_epochs=local_epochs,
                lambda_proto=lambda_proto
            )
            
            # 步骤C: 服务器聚合
            logger.info("步骤C: 服务器聚合...")
            
            # 聚合模型参数
            self.aggregate_models(client_results, client_weights)
            
            # 聚合原型
            self.aggregate_prototypes(client_prototypes, client_weights)
            
            # 计算原型距离 (用于监控)
            prototype_distance = self._calculate_prototype_distance(client_prototypes)
            
            # 记录训练历史
            self.training_history['rounds'].append(round_idx + 1)
            self.training_history['selected_clients'].append(selected_clients)
            self.training_history['prototype_distances'].append(prototype_distance)
            
            round_time = time.time() - round_start_time
            logger.info(f"第 {round_idx + 1} 轮完成，耗时 {round_time:.2f} 秒")
            logger.info(f"原型距离: {prototype_distance:.4f}")
            
            # 定期评估
            if (round_idx + 1) % eval_frequency == 0 or round_idx == num_rounds - 1:
                logger.info(f"正在评估第 {round_idx + 1} 轮...")
                self._evaluate_round(client_manager, client_prototypes, round_idx + 1)
        
        logger.info(f"\n联邦学习训练完成，共 {num_rounds} 轮!")
        
        return {
            'global_model': self.global_model,
            'global_prototype': self.global_prototype,
            'training_history': self.training_history
        }
    
    def _calculate_prototype_distance(self, client_prototypes: Dict[str, torch.Tensor]) -> float:
        """
        计算客户端原型之间的距离
        
        Args:
            client_prototypes: 客户端原型字典
            
        Returns:
            float: 平均原型距离
        """
        if len(client_prototypes) < 2:
            return 0.0
        
        prototypes = list(client_prototypes.values())
        distances = []
        
        for i in range(len(prototypes)):
            for j in range(i + 1, len(prototypes)):
                distance = torch.norm(prototypes[i] - prototypes[j]).item()
                distances.append(distance)
        
        return np.mean(distances)
    
    def _evaluate_round(self, client_manager, client_prototypes: Dict[str, torch.Tensor], round_idx: int) -> None:
        """
        评估当前轮次
        
        Args:
            client_manager: 客户端管理器
            client_prototypes: 客户端原型字典
            round_idx: 轮次索引
        """
        try:
            # 评估所有客户端
            eval_results = client_manager.evaluate_clients(
                global_prototype=self.global_prototype,
                local_prototypes=client_prototypes
            )
            
            # 计算平均AUC
            auc_scores = [result['auc'] for result in eval_results.values() if 'auc' in result]
            if auc_scores:
                avg_auc = np.mean(auc_scores)
                logger.info(f"Round {round_idx} - Average AUC: {avg_auc:.4f}")
                
                # 记录到训练历史
                if 'eval_aucs' not in self.training_history:
                    self.training_history['eval_aucs'] = []
                self.training_history['eval_aucs'].append(avg_auc)
            
        except Exception as e:
            logger.error(f"Error during evaluation at round {round_idx}: {e}")
    
    def save_checkpoint(self, save_path: str) -> None:
        """
        保存检查点
        
        Args:
            save_path: 保存路径
        """
        checkpoint = {
            'global_model_state': self.global_model.state_dict(),
            'global_prototype': self.global_prototype.cpu(),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str) -> None:
        """
        加载检查点
        
        Args:
            load_path: 加载路径
        """
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint['global_model_state'])
        self.global_prototype = checkpoint['global_prototype'].to(self.device)
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Checkpoint loaded from {load_path}")
    
    def get_training_summary(self) -> Dict[str, any]:
        """
        获取训练摘要
        
        Returns:
            Dict[str, any]: 训练摘要
        """
        summary = {
            'total_rounds': len(self.training_history['rounds']),
            'final_prototype_distance': self.training_history['prototype_distances'][-1] if self.training_history['prototype_distances'] else 0.0,
            'client_participation': {}
        }
        
        # 统计客户端参与情况
        for client_id in set().union(*self.training_history['selected_clients']):
            participation_count = sum(1 for round_clients in self.training_history['selected_clients'] if client_id in round_clients)
            summary['client_participation'][client_id] = participation_count
        
        return summary


class ServerManager:
    """
    服务器管理器
    管理服务器和客户端之间的交互
    """
    
    def __init__(self, device: torch.device):
        """
        初始化服务器管理器
        
        Args:
            device: 设备
        """
        self.device = device
        self.server = None
        self.client_manager = None
        
        logger.info(f"ServerManager initialized on device {device}")
    
    def setup_server(self, global_model: nn.Module, **server_kwargs) -> None:
        """
        设置服务器
        
        Args:
            global_model: 全局模型
            **server_kwargs: 服务器参数
        """
        self.server = Server(global_model, self.device, **server_kwargs)
        logger.info("Server setup completed")
    
    def setup_client_manager(self, client_manager) -> None:
        """
        设置客户端管理器
        
        Args:
            client_manager: 客户端管理器
        """
        self.client_manager = client_manager
        logger.info("Client manager setup completed")
    
    def run_training(self, **training_kwargs) -> Dict[str, any]:
        """
        运行训练
        
        Args:
            **training_kwargs: 训练参数
            
        Returns:
            Dict[str, any]: 训练结果
        """
        if self.server is None or self.client_manager is None:
            raise ValueError("Server and client manager must be setup first")
        
        return self.server.run_federation(self.client_manager, **training_kwargs)
    
    def get_server(self) -> Optional[Server]:
        """
        获取服务器
        
        Returns:
            Optional[Server]: 服务器对象
        """
        return self.server


if __name__ == "__main__":
    # 测试服务器聚合循环
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    device = torch.device('cpu')
    feature_dim = 128
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, feature_dim)
        
        def forward(self, x, global_prototype):
            return self.linear(x)
    
    # 创建服务器
    global_model = MockModel()
    server = Server(
        global_model=global_model,
        device=device,
        feature_dim=feature_dim
    )
    
    print(f"Server initialized. Global prototype shape: {server.global_prototype.shape}")
    
    # 测试模型聚合
    print("\nTesting model aggregation...")
    client_models = {
        'client1': {'linear.weight': torch.randn(feature_dim, 128), 'linear.bias': torch.randn(feature_dim)},
        'client2': {'linear.weight': torch.randn(feature_dim, 128), 'linear.bias': torch.randn(feature_dim)},
        'client3': {'linear.weight': torch.randn(feature_dim, 128), 'linear.bias': torch.randn(feature_dim)}
    }
    client_weights = {'client1': 0.4, 'client2': 0.3, 'client3': 0.3}
    
    aggregated_state = server.aggregate_models(client_models, client_weights)
    print(f"Aggregated model state keys: {list(aggregated_state.keys())}")
    
    # 测试原型聚合
    print("\nTesting prototype aggregation...")
    client_prototypes = {
        'client1': torch.randn(feature_dim),
        'client2': torch.randn(feature_dim),
        'client3': torch.randn(feature_dim)
    }
    
    aggregated_prototype = server.aggregate_prototypes(client_prototypes, client_weights)
    print(f"Aggregated prototype shape: {aggregated_prototype.shape}")
    
    # 测试客户端选择
    print("\nTesting client selection...")
    available_clients = ['client1', 'client2', 'client3', 'client4', 'client5']
    selected_clients = server.select_clients(available_clients)
    print(f"Selected clients: {selected_clients}")
    
    print("\nAll tests passed!")
