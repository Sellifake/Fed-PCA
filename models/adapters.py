"""
Adapter Networks for Fed-ProFiLA-AD
实现本地适配器网络和FiLM参数生成器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FiLMGenerator(nn.Module):
    """
    FiLM参数生成器网络 h
    从全局原型生成FiLM调制参数 (γ, β)
    """
    
    def __init__(
        self, 
        input_dim: int = 128, 
        output_dim: int = 128,
        hidden_dims: list = [256, 512]
    ):
        """
        初始化FiLM参数生成器
        
        Args:
            input_dim: 输入维度 (全局原型维度)
            output_dim: 输出维度 (FiLM参数维度，应该是2倍通道数)
            hidden_dims: 隐藏层维度列表
        """
        super(FiLMGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建MLP网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"FiLMGenerator initialized: {input_dim} -> {output_dim}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)  # 小的正偏置
    
    def forward(self, global_prototype: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            global_prototype: 全局原型 [batch_size, input_dim] 或 [input_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (gamma, beta) 调制参数
        """
        # 确保输入是2D张量
        if global_prototype.dim() == 1:
            global_prototype = global_prototype.unsqueeze(0)
        
        # 生成FiLM参数
        film_params = self.network(global_prototype)
        
        # 分割为gamma和beta
        # 假设output_dim是2倍通道数，前半部分是gamma，后半部分是beta
        channels = self.output_dim // 2
        gamma = film_params[:, :channels]
        beta = film_params[:, channels:]
        
        return gamma, beta


class Adapter(nn.Module):
    """
    本地适配器网络 A_i
    使用FiLM参数进行条件调制的瓶颈网络
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """
        初始化适配器网络
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏通道数
            output_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 填充大小
        """
        super(Adapter, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        # 第一个卷积层
        self.conv1 = nn.Conv2d(
            input_channels, 
            hidden_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_channels)
        
        # 激活函数
        self.activation = nn.ReLU(inplace=True)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(
            hidden_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"Adapter initialized: {input_channels} -> {hidden_channels} -> {output_channels}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)  # 小的正偏置
    
    def forward(
        self, 
        x: torch.Tensor, 
        film_params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [batch_size, input_channels, height, width]
            film_params: FiLM参数 (gamma, beta)
            
        Returns:
            torch.Tensor: 调制后的特征图 [batch_size, output_channels, height, width]
        """
        gamma, beta = film_params
        
        # 第一个卷积
        u = self.conv1(x)
        
        # 层归一化 (在通道维度上)
        batch_size, channels, height, width = u.shape
        u_flat = u.view(batch_size, channels, -1)  # [B, C, H*W]
        u_norm = self.layer_norm(u_flat.transpose(1, 2)).transpose(1, 2)  # [B, C, H*W]
        u_norm = u_norm.view(batch_size, channels, height, width)  # [B, C, H, W]
        
        # 激活函数
        u_norm = self.activation(u_norm)
        
        # FiLM调制
        # 扩展gamma和beta到空间维度
        # gamma和beta的形状是 [B, hidden_channels]，需要扩展到 [B, hidden_channels, 1, 1]
        gamma_expanded = gamma.unsqueeze(-1).unsqueeze(-1)  # [B, hidden_channels, 1, 1]
        beta_expanded = beta.unsqueeze(-1).unsqueeze(-1)    # [B, hidden_channels, 1, 1]
        
        # 应用调制: u = u * gamma + beta
        u_modulated = u_norm * gamma_expanded + beta_expanded
        
        # 第二个卷积
        output = self.conv2(u_modulated)
        
        return output


class ResidualAdapter(nn.Module):
    """
    残差适配器网络 (可选的高级版本)
    包含残差连接的适配器网络
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_channels: int = 64,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """
        初始化残差适配器网络
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏通道数
            output_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 填充大小
        """
        super(ResidualAdapter, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        
        # 主路径
        self.conv1 = nn.Conv2d(
            input_channels, 
            hidden_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.activation = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            hidden_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        
        # 残差连接 (如果输入输出通道数不同)
        if input_channels != output_channels:
            self.residual_conv = nn.Conv2d(
                input_channels, 
                output_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0
            )
        else:
            self.residual_conv = None
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"ResidualAdapter initialized: {input_channels} -> {hidden_channels} -> {output_channels}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        film_params: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [batch_size, input_channels, height, width]
            film_params: FiLM参数 (gamma, beta)
            
        Returns:
            torch.Tensor: 调制后的特征图 [batch_size, output_channels, height, width]
        """
        gamma, beta = film_params
        
        # 保存残差连接
        residual = x
        
        # 第一个卷积
        u = self.conv1(x)
        
        # 层归一化
        batch_size, channels, height, width = u.shape
        u_flat = u.view(batch_size, channels, -1)
        u_norm = self.layer_norm(u_flat.transpose(1, 2)).transpose(1, 2)
        u_norm = u_norm.view(batch_size, channels, height, width)
        
        # 激活函数
        u_norm = self.activation(u_norm)
        
        # FiLM调制
        gamma_expanded = gamma.unsqueeze(-1).unsqueeze(-1)
        beta_expanded = beta.unsqueeze(-1).unsqueeze(-1)
        u_modulated = u_norm * gamma_expanded + beta_expanded
        
        # 第二个卷积
        output = self.conv2(u_modulated)
        
        # 残差连接
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
        
        output = output + residual
        
        return output


if __name__ == "__main__":
    # 测试适配器网络
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试参数
    batch_size = 4
    input_channels = 1
    hidden_channels = 32
    output_channels = 64
    height, width = 128, 32  # Mel频谱图尺寸
    prototype_dim = 128
    
    # 创建测试数据
    x = torch.randn(batch_size, input_channels, height, width)
    global_prototype = torch.randn(prototype_dim)
    
    # 测试FiLM生成器
    print("Testing FiLMGenerator...")
    film_generator = FiLMGenerator(
        input_dim=prototype_dim,
        output_dim=hidden_channels * 2  # gamma和beta各占一半
    )
    
    gamma, beta = film_generator(global_prototype)
    print(f"Gamma shape: {gamma.shape}, Beta shape: {beta.shape}")
    
    # 测试适配器
    print("\nTesting Adapter...")
    adapter = Adapter(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=output_channels
    )
    
    output = adapter(x, (gamma, beta))
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    # 测试残差适配器
    print("\nTesting ResidualAdapter...")
    residual_adapter = ResidualAdapter(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=output_channels
    )
    
    output_residual = residual_adapter(x, (gamma, beta))
    print(f"Input shape: {x.shape}, Output shape: {output_residual.shape}")
    
    print("\nAll tests passed!")
