"""
Client Training Loop for Fed-ProFiLA-AD (基础版本)
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
    联邦学习客户端类 (基础版本)
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
        
        # 将模型移动到设备
        self.model = self.model.to(device)
        
        # 设置优化器
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)
        
        # 训练历史
        self.training_history = {
            'losses': [],
            'task_losses': [],
            'proto_losses': [],
            'sep_losses': [],
            'supcon_losses': [],
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
    
    def _compute_local_prototype(self, global_prototype: torch.Tensor) -> torch.Tensor:
        """
        计算本地原型
        
        Args:
            global_prototype: 全局原型
            
        Returns:
            torch.Tensor: 本地原型
        """
        return compute_local_prototype(
            model=self.model,
            dataloader=self.train_loader,
            global_prototype=global_prototype,
            device=self.device
        )
    
    def train(
        self,
        global_theta: Dict[str, torch.Tensor],
        global_mu: torch.Tensor,
        local_epochs: int = 5,
        lambda_proto: float = 0.1,
        lambda_contrastive: float = 0.5,
        contrastive_margin: float = 0.8,
        lambda_separation: float = 0.5,
        separation_margin: float = 0.8,
        lambda_supcon: float = 1.0,
        temperature: float = 0.2
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
        # 确保模型在训练模式
        self.model.train()
        
        # 步骤B.i: 接收全局参数
        self._load_shared_state(global_theta)
        
        # 步骤B.ii: 计算FiLM参数 (在模型前向传播时完成)
        
        # 数据与类分布日志
        try:
            labels = getattr(self.train_loader.dataset, 'labels', None)
            if labels is not None:
                num_norm = sum(1 for y in labels if y == 0)
                num_abn = sum(1 for y in labels if y == 1)
                logger.info(f"Client {self.client_id} data: total={len(labels)}, normal={num_norm}, abnormal={num_abn}")
        except Exception:
            pass

        # 步骤B.iii: 计算初始本地原型（用于参考，但不完全冻结）
        mu_local_init = self._compute_local_prototype(global_mu)
        
        # 步骤B.iv: 本地训练轮次
        self.model.train()
        epoch_losses = []
        epoch_task_losses = []
        epoch_proto_losses = []
        epoch_sep_losses = []
        epoch_supcon_losses = []
        
        # 创建进度条（简化显示）
        pbar = tqdm(range(local_epochs), desc=f"Client {self.client_id}", leave=False, disable=True)
        
        for epoch in pbar:
            epoch_loss = 0.0
            epoch_task_loss = 0.0
            epoch_proto_loss = 0.0
            num_batches = 0
            grad_norm_accum = 0.0
            
            # 创建批次进度条（不显示）
            batch_pbar = self.train_loader
            
            # 在每个epoch开始时，更新本地原型（使用当前模型状态）
            if epoch > 0:
                with torch.no_grad():
                    mu_local_updated = self._compute_local_prototype(global_mu)
                    # 使用移动平均平滑原型更新，避免突变
                    mu_local_init = 0.8 * mu_local_init + 0.2 * mu_local_updated
            
            # 初始化当前epoch的本地原型
            mu_local_epoch = mu_local_init.clone().to(self.device)
            
            for batch_x, batch_y in batch_pbar:
                batch_x = batch_x.to(self.device)
                global_mu = global_mu.to(self.device)
                
                # 前向传播（使用全局原型）
                z_batch = self.model(batch_x, global_mu)
                
                # 计算损失（使用动态本地原型，结合当前epoch原型和批次均值）
                # 这样可以更好地适应批次数据分布，同时保持稳定性
                batch_mean = torch.mean(z_batch, dim=0)
                mu_local_batch = 0.95 * mu_local_epoch + 0.05 * batch_mean
                
                # 将标签移至设备
                y_batch = batch_y.to(self.device) if isinstance(batch_y, torch.Tensor) else torch.tensor(batch_y, device=self.device)

                # 原型正则预热：从0线性增至目标lambda_proto
                lambda_proto_eff = float(lambda_proto) * float(epoch + 1) / float(max(1, local_epochs))

                total_loss, task_loss, proto_loss, contrastive_loss, separation_loss, supcon_loss = compute_total_loss(
                    z_batch=z_batch,
                    mu_local=mu_local_batch,
                    mu_global=global_mu,
                    lambda_proto=lambda_proto_eff,
                    y_batch=y_batch,
                    lambda_contrastive=lambda_contrastive,
                    contrastive_margin=contrastive_margin,
                    lambda_separation=lambda_separation,
                    separation_margin=separation_margin,
                    lambda_supcon=lambda_supcon,
                    temperature=temperature
                )
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度范数统计
                total_grad_sq = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        g = p.grad.data
                        total_grad_sq += float(torch.sum(g * g).item())
                grad_norm = total_grad_sq ** 0.5
                grad_norm_accum += grad_norm
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 更新当前epoch的本地原型（用于下一个batch，但不影响梯度）
                with torch.no_grad():
                    # 使用更保守的更新策略
                    mu_local_epoch = 0.9 * mu_local_epoch + 0.1 * batch_mean
                
                # 记录损失
                epoch_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_proto_loss += proto_loss.item()
                epoch_sep_loss = separation_loss.item()
                epoch_sep_losses.append(epoch_sep_loss)
                epoch_supcon_losses.append(supcon_loss.item())
                num_batches += 1
                
            
            # 计算平均损失
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_task_loss = epoch_task_loss / num_batches if num_batches > 0 else 0.0
            avg_proto_loss = epoch_proto_loss / num_batches if num_batches > 0 else 0.0
            avg_grad_norm = grad_norm_accum / num_batches if num_batches > 0 else 0.0
            avg_sep_loss = np.mean(epoch_sep_losses) if epoch_sep_losses else 0.0
            avg_supcon_loss = np.mean(epoch_supcon_losses) if epoch_supcon_losses else 0.0
            
            epoch_losses.append(avg_loss)
            epoch_task_losses.append(avg_task_loss)
            epoch_proto_losses.append(avg_proto_loss)
            self.training_history['sep_losses'].append(avg_sep_loss)
            self.training_history['supcon_losses'].append(avg_supcon_loss)
            
            # 记录分离项
            
            
            # 记录每个epoch的概览
            try:
                with torch.no_grad():
                    mu_norm = torch.nn.functional.normalize(mu_local_epoch, dim=0)
                    mu_g_norm = torch.nn.functional.normalize(global_mu, dim=0)
                    mu_dist = torch.norm(mu_norm - mu_g_norm).item()
                logger.info(
                    f"Client {self.client_id} | epoch {epoch+1}/{local_epochs} | "
                    f"loss {avg_loss:.4f} (task {avg_task_loss:.4f}, proto {avg_proto_loss:.4f}, sep {avg_sep_loss:.4f}, sup {avg_supcon_loss:.4f}) | "
                    f"grad_norm {avg_grad_norm:.4f} | mu_dist {mu_dist:.4f}"
                )
            except Exception:
                pass
            
        pbar.close()
        
        # 步骤B.v: 计算新的本地原型
        mu_local_new = self._compute_local_prototype(global_mu)
        
        # 记录训练历史
        self.training_history['losses'].extend(epoch_losses)
        self.training_history['task_losses'].extend(epoch_task_losses)
        self.training_history['proto_losses'].extend(epoch_proto_losses)
        self.training_history['local_prototypes'].append(mu_local_new.cpu().clone())
        
        # 步骤B.vi: 返回更新后的模型参数和新的本地原型
        updated_theta = self._extract_shared_state_dict()
        
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
        mu_local_for_eval = self._compute_local_prototype(global_mu)
        
        # 3. 使用正确的参数调用评估
        return evaluate_client(
            model=self.model,
            test_loader=self.test_loader,
            global_prototype=global_mu,
            local_prototype=mu_local_for_eval,
            device=self.device,
            client_id=self.client_id
        )


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
        # 打印该客户端样本与类分布
        try:
            labels = getattr(client.train_loader.dataset, 'labels', None)
            if labels is not None:
                num_norm = sum(1 for y in labels if y == 0)
                num_abn = sum(1 for y in labels if y == 1)
                logger.info(f"Register client {client.client_id}: train_size={data_size}, normal={num_norm}, abnormal={num_abn}")
        except Exception:
            pass
    
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
        lambda_proto: float = 0.1,
        lambda_contrastive: float = 0.5,
        contrastive_margin: float = 0.8,
        lambda_separation: float = 0.5,
        separation_margin: float = 0.8,
        lambda_supcon: float = 1.0,
        temperature: float = 0.2
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
        
        for client_id in selected_clients:
            if client_id in self.clients:
                client = self.clients[client_id]
                
                # 训练客户端
                updated_theta, new_prototype = client.train(
                    global_theta=global_theta,
                    global_mu=global_mu,
                    local_epochs=local_epochs,
                    lambda_proto=lambda_proto,
                    lambda_contrastive=lambda_contrastive,
                    contrastive_margin=contrastive_margin,
                    lambda_separation=lambda_separation,
                    separation_margin=separation_margin,
                    lambda_supcon=lambda_supcon,
                    temperature=temperature
                )
                
                client_results[client_id] = updated_theta
                client_prototypes[client_id] = new_prototype
            else:
                logger.warning(f"Client {client_id} not found")
        
        return client_results, client_prototypes
    
    def evaluate_clients(
        self,
        global_theta: Dict[str, torch.Tensor],
        global_mu: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """
        评估所有客户端
        
        Args:
            global_theta: 全局模型状态
            global_mu: 全局原型
            
        Returns:
            Dict[str, Dict[str, float]]: 客户端评估结果
        """
        results = {}
        
        for client_id, client in self.clients.items():
            result = client.evaluate(global_theta, global_mu)
            results[client_id] = result
        
        return results
