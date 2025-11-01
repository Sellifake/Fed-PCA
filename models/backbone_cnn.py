"""
Backbone CNN Network for Fed-ProFiLA-AD (基础版本)
实现共享骨干网络，包含FiLM生成器和适配器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import logging

from models.adapters import FiLMGenerator, Adapter

logger = logging.getLogger(__name__)


class BackboneCNN(nn.Module):
    """
    共享骨干网络 f_θ (基础版本)
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
        dropout_rate: float = 0.1,
        use_projection_head: bool = True,
        projection_hidden_dim: int = 256
    ):
        """
        初始化骨干网络
        
        Args:
            input_channels: 输入通道数
            feature_dim: 输出特征维度
            hidden_dims: 隐藏层维度列表
            adapter_hidden_channels: 适配器隐藏通道数
            adapter_output_channels: 适配器输出通道数
            prototype_dim: 全局原型维度（用于FiLM生成器输入）
            dropout_rate: Dropout比率
        """
        super(BackboneCNN, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.hidden_dims = hidden_dims
        self.prototype_dim = prototype_dim
        
        # FiLM参数生成器 h (从全局原型生成)
        self.film_generator = FiLMGenerator(
            input_dim=prototype_dim,  # 从全局原型输入
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

        # 投影头：提升对比学习判别力
        self.use_projection_head = use_projection_head
        if use_projection_head:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, projection_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_hidden_dim, feature_dim),
                nn.LayerNorm(feature_dim)
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
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
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
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, global_prototype: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入频谱图 [batch_size, input_channels, height, width]
            global_prototype: 全局原型 [prototype_dim]
            
        Returns:
            torch.Tensor: 特征嵌入 [batch_size, feature_dim]
        """
        # 1. 从全局原型生成FiLM参数
        gamma, beta = self.film_generator(global_prototype)
        
        # 2. 适配器处理（使用FiLM参数调制）
        u = self.adapter(x, (gamma, beta))
        
        # 3. 主编码器
        z = self.encoder(u)
        if self.use_projection_head:
            z = self.projection(z)
        # 输出L2归一化的embedding
        z = F.normalize(z, p=2.0, dim=-1)
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
        backbone_type: 骨干网络类型 ("default")
        **kwargs: 传递给骨干网络的参数
        
    Returns:
        nn.Module: 骨干网络
    """
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
    
    print("\nAll tests passed!")
