"""
Backbone CNN Network for Fed-ProFiLA-AD
实现共享骨干网络，包含FiLM生成器和适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

from models.adapters import FiLMGenerator, Adapter
from models.device_aware_adapters import DeviceTypeEncoder

logger = logging.getLogger(__name__)


class BackboneCNN(nn.Module):
    """
    共享骨干网络 f_θ
    包含FiLM生成器、适配器和主编码器
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        hidden_dims: list = [64, 128, 256],
        adapter_hidden_channels: int = 32,
        adapter_output_channels: int = 64,
        prototype_dim: int = 128,
        device_type_dim: int = 16,
        dropout_rate: float = 0.1
    ):
        """
        初始化骨干网络
        
        Args:
            input_channels: 输入通道数
            feature_dim: 输出特征维度
            hidden_dims: 隐藏层维度列表
            adapter_hidden_channels: 适配器隐藏通道数
            adapter_output_channels: 适配器输出通道数
            prototype_dim: 全局原型维度（用于损失计算，不作为输入）
            device_type_dim: 设备类型嵌入维度
            dropout_rate: Dropout比率
        """
        super(BackboneCNN, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.prototype_dim = prototype_dim
        self.device_type_dim = device_type_dim
        
        # 设备类型编码器
        self.device_type_encoder = DeviceTypeEncoder(device_type_dim)
        
        # FiLM参数生成器 h（输入改为设备类型嵌入）
        self.film_generator = FiLMGenerator(
            input_dim=device_type_dim,  # 改为使用设备类型维度
            output_dim=adapter_hidden_channels * 2,  # gamma和beta各占一半
            hidden_dims=[256, 512]
        )
        
        # 本地适配器 A_i
        self.adapter = Adapter(
            input_channels=input_channels,
            hidden_channels=adapter_hidden_channels,
            output_channels=adapter_output_channels
        )
        
        # 主编码器网络
        self.encoder = self._build_encoder(
            input_channels=adapter_output_channels,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"BackboneCNN initialized: {input_channels} -> {feature_dim}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def _build_encoder(self, input_channels: int, feature_dim: int, 
                      hidden_dims: list, dropout_rate: float) -> nn.Module:
        """
        构建主编码器网络
        
        Args:
            input_channels: 输入通道数
            feature_dim: 输出特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout比率
            
        Returns:
            nn.Module: 编码器网络
        """
        layers = []
        prev_channels = input_channels
        
        # 卷积层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Conv2d(prev_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate)
            ])
            prev_channels = hidden_dim
        
        # 全局平均池化
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        
        # 展平
        layers.append(nn.Flatten())
        
        # 最终全连接层
        layers.extend([
            nn.Linear(prev_channels, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, device_type_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播（修复后：打破输入即目标的反馈循环）
        
        Args:
            x: 输入频谱图 [batch_size, input_channels, height, width]
            device_type_id: 设备类型ID [batch_size] 或标量
            
        Returns:
            torch.Tensor: 特征嵌入 [batch_size, feature_dim]
        """
        # 1. 获取设备类型嵌入（替代global_prototype作为输入）
        device_embedding = self.device_type_encoder(device_type_id)
        
        # 2. 用设备类型嵌入生成FiLM参数
        gamma, beta = self.film_generator(device_embedding)
        
        # 3. 适配器处理
        u = self.adapter(x, (gamma, beta))
        
        # 4. 主编码器
        z = self.encoder(u)
        
        return z
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_dim
    
    def get_prototype_dim(self) -> int:
        """获取原型维度"""
        return self.prototype_dim


class DCASEBackbone(nn.Module):
    """
    DCASE挑战赛风格的骨干网络
    基于DCASE 2020 Task 2的架构
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 128,
        prototype_dim: int = 128,
        device_type_dim: int = 16,
        dropout_rate: float = 0.1
    ):
        """
        初始化DCASE骨干网络
        
        Args:
            input_channels: 输入通道数
            feature_dim: 输出特征维度
            prototype_dim: 全局原型维度（用于损失计算，不作为输入）
            device_type_dim: 设备类型嵌入维度
            dropout_rate: Dropout比率
        """
        super(DCASEBackbone, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.prototype_dim = prototype_dim
        self.device_type_dim = device_type_dim
        
        # 设备类型编码器
        self.device_type_encoder = DeviceTypeEncoder(device_type_dim)
        
        # FiLM参数生成器（输入改为设备类型嵌入，而不是global_prototype）
        self.film_generator = FiLMGenerator(
            input_dim=device_type_dim,  # 改为使用设备类型维度
            output_dim=64,  # 32 * 2 for gamma and beta
            hidden_dims=[256, 512]
        )
        
        # 适配器
        self.adapter = Adapter(
            input_channels=input_channels,
            hidden_channels=32,
            output_channels=64
        )
        
        # DCASE风格的编码器
        self.encoder = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # 第二个卷积块
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # 第三个卷积块
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            # --- 修改开始：添加BatchNorm1d防止坍塌 ---
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),  # <-- 在这里添加BN防止坍塌
            nn.ReLU(inplace=True)         # <-- 确保BN在ReLU之前
            # --- 修改结束 ---
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"DCASEBackbone initialized: {input_channels} -> {feature_dim}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, device_type_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播（修复后：打破输入即目标的反馈循环）
        
        Args:
            x: 输入频谱图 [batch_size, input_channels, height, width]
            device_type_id: 设备类型ID [batch_size] 或标量
            
        Returns:
            torch.Tensor: 特征嵌入 [batch_size, feature_dim]
        """
        # 1. 获取设备类型嵌入（替代global_prototype作为输入）
        device_embedding = self.device_type_encoder(device_type_id)
        
        # 2. 用设备类型嵌入生成FiLM参数
        gamma, beta = self.film_generator(device_embedding)
        
        # 3. 适配器处理
        u = self.adapter(x, (gamma, beta))
        
        # 4. 主编码器
        z = self.encoder(u)
        
        return z
    
    def get_feature_dim(self) -> int:
        """获取特征维度"""
        return self.feature_dim
    
    def get_prototype_dim(self) -> int:
        """获取原型维度"""
        return self.prototype_dim


def create_backbone(backbone_type: str = "default", **kwargs) -> nn.Module:
    """
    创建骨干网络
    
    Args:
        backbone_type: 骨干网络类型 ("default" 或 "dcase")
        **kwargs: 传递给骨干网络的参数
        
    Returns:
        nn.Module: 骨干网络
    """
    if backbone_type.lower() == "dcase":
        # DCASEBackbone只接受特定参数
        dcase_kwargs = {
            'input_channels': kwargs.get('input_channels', 1),
            'feature_dim': kwargs.get('feature_dim', 128),
            'prototype_dim': kwargs.get('prototype_dim', 128),
            'dropout_rate': kwargs.get('dropout_rate', 0.1)
        }
        return DCASEBackbone(**dcase_kwargs)
    else:
        return BackboneCNN(**kwargs)


if __name__ == "__main__":
    # 测试骨干网络
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试参数
    batch_size = 4
    input_channels = 1
    height, width = 128, 32  # Mel频谱图尺寸
    feature_dim = 128
    prototype_dim = 128
    
    # 创建测试数据
    x = torch.randn(batch_size, input_channels, height, width)
    global_prototype = torch.randn(prototype_dim)
    
    # 测试默认骨干网络
    print("Testing BackboneCNN...")
    backbone = BackboneCNN(
        input_channels=input_channels,
        feature_dim=feature_dim,
        prototype_dim=prototype_dim
    )
    
    output = backbone(x, global_prototype)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    print(f"Feature dim: {backbone.get_feature_dim()}, Prototype dim: {backbone.get_prototype_dim()}")
    
    # 测试DCASE骨干网络
    print("\nTesting DCASEBackbone...")
    dcase_backbone = DCASEBackbone(
        input_channels=input_channels,
        feature_dim=feature_dim,
        prototype_dim=prototype_dim
    )
    
    output_dcase = dcase_backbone(x, global_prototype)
    print(f"Input shape: {x.shape}, Output shape: {output_dcase.shape}")
    print(f"Feature dim: {dcase_backbone.get_feature_dim()}, Prototype dim: {dcase_backbone.get_prototype_dim()}")
    
    # 测试工厂函数
    print("\nTesting create_backbone...")
    backbone_factory = create_backbone("default", input_channels=input_channels, feature_dim=feature_dim)
    output_factory = backbone_factory(x, global_prototype)
    print(f"Factory backbone output shape: {output_factory.shape}")
    
    print("\nAll tests passed!")
