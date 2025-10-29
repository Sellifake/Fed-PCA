"""
Client Training Loop for Fed-ProFiLA-AD
实现客户端本地训练循环
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import time
from tqdm import tqdm

from methods.fed_profila_ad import (
    compute_total_loss,
    compute_local_prototype
)

logger = logging.getLogger(__name__)


class Client:
    """
    联邦学习客户端类
    实现Fed-ProFiLA-AD的客户端训练逻辑
    """
    
    SHARED_PREFIXES = ("film_generator", "encoder")

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        weight_decay: float = 1e-4
    ):
        """
        初始化客户端
        
        Args:
            client_id: 客户端ID
            model: 模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            device: 设备
            learning_rate: 学习率
            optimizer_name: 优化器名称
            weight_decay: 权重衰减
        """
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # 获取客户端的设备类型ID
        from models.device_aware_adapters import get_device_type_id
        self.device_type_id = get_device_type_id(client_id)
        
        # 将模型移动到设备
        self.model = self.model.to(device)
        
        # 设置优化器
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)
        
        # 训练历史
        self.training_history = {
            'losses': [],
            'task_losses': [],
            'proto_losses': [],
            'local_prototypes': []
        }
        
        logger.info(f"Client {client_id} initialized on device {device}")
    
    def _is_shared_parameter(self, name: str) -> bool:
        """判断参数是否属于共享部分"""
        return name.startswith(Client.SHARED_PREFIXES)

    def _extract_shared_state_dict(
        self,
        state_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """提取共享参数的状态字典"""
        if state_dict is None:
            state_dict = self.model.state_dict()
        return {
            name: param.clone()
            for name, param in state_dict.items()
            if self._is_shared_parameter(name)
        }

    def _load_shared_state(self, shared_state: Dict[str, torch.Tensor]) -> None:
        """仅加载共享参数状态，保护本地适配器参数"""
        if not shared_state:
            return
        filtered_state = {
            name: param
            for name, param in shared_state.items()
            if self._is_shared_parameter(name)
        }
        if not filtered_state:
            return
        self.model.load_state_dict(filtered_state, strict=False)

    def _create_optimizer(self, optimizer_name: str, learning_rate: float, weight_decay: float) -> optim.Optimizer:
        """
        创建优化器
        
        Args:
            optimizer_name: 优化器名称
            learning_rate: 学习率
            weight_decay: 权重衰减
            
        Returns:
            optim.Optimizer: 优化器
        """
        if optimizer_name.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name.lower() == "adamw":
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _compute_local_prototype(self, global_prototype: torch.Tensor, device_type_id: int = None) -> torch.Tensor:
        """
        计算本地原型
        
        Args:
            global_prototype: 全局原型（仅用于损失计算）
            device_type_id: 设备类型ID
            
        Returns:
            torch.Tensor: 本地原型
        """
        if device_type_id is None:
            device_type_id = self.device_type_id
        return compute_local_prototype(
            model=self.model,
            dataloader=self.train_loader,
            global_prototype=global_prototype,
            device=self.device,
            device_type_id=device_type_id
        )
    
    def train(
        self,
        global_theta: Dict[str, torch.Tensor],
        global_mu: torch.Tensor,
        local_epochs: int = 5,
        lambda_proto: float = 0.1
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        客户端本地训练
        
        Args:
            global_theta: 全局模型参数
            global_mu: 全局原型
            local_epochs: 本地训练轮数
            lambda_proto: 原型对齐损失权重
            
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: (更新后的模型参数, 新的本地原型)
        """
        logger.info(f"客户端 {self.client_id} 开始本地训练...")
        
        # 确保模型在训练模式（防止BatchNorm等冻结）
        self.model.train()
        
        # 步骤B.i: 接收全局参数
        self._load_shared_state(global_theta)
        
        # 步骤B.ii: 计算FiLM参数 (在模型内部完成)
        
        # 步骤B.iii: 计算并冻结本地原型
        logger.info(f"正在计算客户端 {self.client_id} 的本地原型...")
        mu_local_frozen = self._compute_local_prototype(global_mu, self.device_type_id).detach()
        
        # 步骤B.iv: 本地训练轮次
        self.model.train()
        epoch_losses = []
        epoch_task_losses = []
        epoch_proto_losses = []
        
        # 创建进度条
        pbar = tqdm(range(local_epochs), desc=f"Client {self.client_id} Training", leave=False)
        
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_proto_loss = 0.0
            num_batches = 0
            
            # 创建批次进度条
            batch_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{local_epochs}", leave=False)
            
            for batch_data in batch_pbar:
                # 解包数据（可能是2个或3个值）
                if len(batch_data) == 3:
                    batch_x, batch_y, _ = batch_data  # 忽略device_type
                else:
                    batch_x, batch_y = batch_data
                batch_x = batch_x.to(self.device)
                global_mu = global_mu.to(self.device)
                mu_local_frozen = mu_local_frozen.to(self.device)
                
                # 前向传播（使用设备类型ID而不是global_mu）
                device_type_id_tensor = torch.tensor([self.device_type_id] * batch_x.size(0), 
                                                     dtype=torch.long, device=self.device)
                z_batch = self.model(batch_x, device_type_id_tensor)
                
                # 添加调试信息
                if epoch == 0 and num_batches == 0:  # 只在第一个epoch的第一个batch打印
                    logger.info(f"客户端 {self.client_id} 调试信息:")
                    logger.info(f"  输入批次形状: {batch_x.shape}, 数值范围: [{batch_x.min().item():.4f}, {batch_x.max().item():.4f}]")
                    logger.info(f"  特征输出形状: {z_batch.shape}, 数值范围: [{z_batch.min().item():.4f}, {z_batch.max().item():.4f}]")
                    logger.info(f"  特征均值: {z_batch.mean().item():.4f}, 特征方差: {z_batch.var().item():.4f}")
                    logger.info(f"  全局原型形状: {global_mu.shape}, 数值范围: [{global_mu.min().item():.4f}, {global_mu.max().item():.4f}]")
                    logger.info(f"  本地原型形状: {mu_local_frozen.shape}, 数值范围: [{mu_local_frozen.min().item():.4f}, {mu_local_frozen.max().item():.4f}]")
                    logger.info(f"  原型范数: {torch.norm(mu_local_frozen).item():.4f}")
                
                # 计算损失
                total_loss, task_loss, proto_loss = compute_total_loss(
                    z_batch=z_batch,
                    mu_local=mu_local_frozen,
                    mu_global=global_mu,
                    lambda_proto=lambda_proto
                )
                
                # 添加损失调试信息
                if epoch == 0 and num_batches == 0:  # 只在第一个epoch的第一个batch打印
                    logger.info(f"  总损失: {total_loss.item():.6f}")
                    logger.info(f"  任务损失: {task_loss.item():.6f}")
                    logger.info(f"  原型损失: {proto_loss.item():.6f}")
                    logger.info(f"  原型权重: {lambda_proto}")
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # 记录损失
                epoch_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_proto_loss += proto_loss.item()
                num_batches += 1
                
                # 更新批次进度条
                batch_pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Task': f'{task_loss.item():.4f}',
                    'Proto': f'{proto_loss.item():.4f}'
                })
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches
            avg_task_loss = epoch_task_loss / num_batches
            avg_proto_loss = epoch_proto_loss / num_batches
            
            epoch_losses.append(avg_loss)
            epoch_task_losses.append(avg_task_loss)
            epoch_proto_losses.append(avg_proto_loss)
            
            # 更新主进度条
            pbar.set_postfix({
                'Avg Loss': f'{avg_loss:.4f}',
                'Task': f'{avg_task_loss:.4f}',
                'Proto': f'{avg_proto_loss:.4f}'
            })
            
            logger.info(f"客户端 {self.client_id} - 轮次 {epoch+1}/{local_epochs}: "
                       f"总损失={avg_loss:.4f}, 任务损失={avg_task_loss:.4f}, 原型损失={avg_proto_loss:.4f}")
        
        pbar.close()
        
        # 步骤B.v: 计算新的本地原型
        logger.info(f"正在计算客户端 {self.client_id} 的新本地原型...")
        mu_local_new = self._compute_local_prototype(global_mu, self.device_type_id)
        
        # 记录训练历史
        self.training_history['losses'].extend(epoch_losses)
        self.training_history['task_losses'].extend(epoch_task_losses)
        self.training_history['proto_losses'].extend(epoch_proto_losses)
        self.training_history['local_prototypes'].append(mu_local_new.cpu().clone())
        
        # 步骤B.vi: 返回更新后的模型参数和新的本地原型
        updated_theta = self._extract_shared_state_dict()
        
        logger.info(f"客户端 {self.client_id} 训练完成")
        
        return updated_theta, mu_local_new
    
    def evaluate(self, global_theta: Dict[str, torch.Tensor], global_mu: torch.Tensor) -> Dict[str, float]:
        """
        评估客户端模型
        
        Args:
            global_theta: 全局模型状态
            global_mu: 全局原型
            
        Returns:
            Dict[str, float]: 评估指标
        """
        from eval.inference import evaluate_client
        
        # 1. 加载收到的全局模型
        self._load_shared_state(global_theta)
        self.model.eval()
        
        # 2. 用加载的模型和训练数据计算评估基准（本地原型）
        mu_local_for_eval = self._compute_local_prototype(global_mu, self.device_type_id)
        
        # 3. 使用正确的参数调用评估
        return evaluate_client(
            model=self.model,
            test_loader=self.test_loader,
            global_prototype=global_mu,
            local_prototype=mu_local_for_eval,
            device=self.device,
            client_id=self.client_id,
            device_type_id=self.device_type_id
        )
    
    def get_training_history(self) -> Dict[str, list]:
        """
        获取训练历史
        
        Returns:
            Dict[str, list]: 训练历史字典
        """
        return self.training_history
    
    def reset_training_history(self) -> None:
        """重置训练历史"""
        self.training_history = {
            'losses': [],
            'task_losses': [],
            'proto_losses': [],
            'local_prototypes': []
        }
    
    def get_model_state(self) -> Dict[str, torch.Tensor]:
        """
        获取模型状态
        
        Returns:
            Dict[str, torch.Tensor]: 模型状态字典
        """
        return self._extract_shared_state_dict()
    
    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        加载模型状态
        
        Args:
            state_dict: 模型状态字典
        """
        self._load_shared_state(state_dict)
    
    def get_local_prototype(self, global_prototype: torch.Tensor) -> torch.Tensor:
        """
        获取当前本地原型
        
        Args:
            global_prototype: 全局原型
            
        Returns:
            torch.Tensor: 本地原型
        """
        return self._compute_local_prototype(global_prototype)


class ClientManager:
    """
    客户端管理器
    管理多个客户端的训练和评估
    """
    
    def __init__(self, device: torch.device):
        """
        初始化客户端管理器
        
        Args:
            device: 设备
        """
        self.device = device
        self.clients = {}
        self.client_weights = {}
        
        logger.info(f"ClientManager initialized on device {device}")
    
    def add_client(self, client: Client) -> None:
        """
        添加客户端
        
        Args:
            client: 客户端对象
        """
        self.clients[client.client_id] = client
        
        # 计算客户端权重 (基于数据量)
        data_size = len(client.train_loader.dataset)
        self.client_weights[client.client_id] = data_size
        
        logger.info(f"Client {client.client_id} added with data size {data_size}")
    
    def get_client(self, client_id: str) -> Optional[Client]:
        """
        获取客户端
        
        Args:
            client_id: 客户端ID
            
        Returns:
            Optional[Client]: 客户端对象
        """
        return self.clients.get(client_id)
    
    def get_all_clients(self) -> Dict[str, Client]:
        """
        获取所有客户端
        
        Returns:
            Dict[str, Client]: 客户端字典
        """
        return self.clients
    
    def get_client_weights(self) -> Dict[str, float]:
        """
        获取客户端权重
        
        Returns:
            Dict[str, float]: 客户端权重字典
        """
        total_weight = sum(self.client_weights.values())
        normalized_weights = {
            client_id: weight / total_weight
            for client_id, weight in self.client_weights.items()
        }
        return normalized_weights
    
    def train_clients(
        self,
        global_theta: Dict[str, torch.Tensor],
        global_mu: torch.Tensor,
        selected_clients: list = None,
        local_epochs: int = 5,
        lambda_proto: float = 0.1
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:
        """
        训练选中的客户端
        
        Args:
            global_theta: 全局模型参数
            global_mu: 全局原型
            selected_clients: 选中的客户端ID列表
            local_epochs: 本地训练轮数
            lambda_proto: 原型对齐损失权重
            
        Returns:
            Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]: (客户端模型参数, 客户端原型)
        """
        if selected_clients is None:
            selected_clients = list(self.clients.keys())
        
        client_results = {}
        client_prototypes = {}
        
        logger.info(f"Training {len(selected_clients)} clients...")
        
        for client_id in selected_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # 训练客户端
                updated_theta, new_prototype = client.train(
                    global_theta=global_theta,
                    global_mu=global_mu,
                    local_epochs=local_epochs,
                    lambda_proto=lambda_proto
                )
                
                client_results[client_id] = updated_theta
                client_prototypes[client_id] = new_prototype
                
                logger.info(f"Client {client_id} training completed")
            else:
                logger.warning(f"Client {client_id} not found")
        
        return client_results, client_prototypes
    
    def evaluate_clients(
        self,
        global_prototype: torch.Tensor,
        local_prototypes: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, float]]:
        """
        评估所有客户端
        
        Args:
            global_prototype: 全局原型
            local_prototypes: 客户端本地原型字典
            
        Returns:
            Dict[str, Dict[str, float]]: 客户端评估结果
        """
        results = {}
        
        for client_id, client in self.clients.items():
            if client_id in local_prototypes:
                result = client.evaluate(global_prototype, local_prototypes[client_id])
                results[client_id] = result
        
        return results


if __name__ == "__main__":
    # 测试客户端训练循环
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据
    device = torch.device('cpu')
    batch_size = 4
    n_mels = 128
    time_frames = 32
    feature_dim = 128
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(n_mels * time_frames, feature_dim)
        
        def forward(self, x, global_prototype):
            x = x.view(x.size(0), -1)
            return self.linear(x)
    
    # 创建模拟数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    x_train = torch.randn(100, 1, n_mels, time_frames)
    y_train = torch.zeros(100)  # 训练时只有Normal数据
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    x_test = torch.randn(50, 1, n_mels, time_frames)
    y_test = torch.randint(0, 2, (50,))  # 测试时有Normal和Abnormal数据
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建客户端
    model = MockModel()
    client = Client(
        client_id="test_client",
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device
    )
    
    # 测试训练
    print("Testing client training...")
    global_theta = model.state_dict()
    global_mu = torch.randn(feature_dim)
    
    updated_theta, new_prototype = client.train(
        global_theta=global_theta,
        global_mu=global_mu,
        local_epochs=2,
        lambda_proto=0.1
    )
    
    print(f"Training completed. New prototype shape: {new_prototype.shape}")
    
    # 测试客户端管理器
    print("\nTesting client manager...")
    manager = ClientManager(device)
    manager.add_client(client)
    
    # 测试训练多个客户端
    client_results, client_prototypes = manager.train_clients(
        global_theta=global_theta,
        global_mu=global_mu,
        local_epochs=1,
        lambda_proto=0.1
    )
    
    print(f"Manager training completed. {len(client_results)} clients trained")
    
    print("\nAll tests passed!")
