"""
Device-Aware Adapters for Cross-Device Federated Learning
实现设备类型感知的适配器网络，支持跨设备联邦学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import logging

from models.adapters import FiLMGenerator, Adapter

logger = logging.getLogger(__name__)


class DeviceTypeEncoder(nn.Module):
    """
    设备类型编码器
    将设备类型信息编码为特征向量
    """
    
    def __init__(self, device_type_dim: int = 16):
        """
        初始化设备类型编码器
        
        Args:
            device_type_dim: 设备类型特征维度
        """
        super(DeviceTypeEncoder, self).__init__()
        
        self.device_type_dim = device_type_dim
        
        # 设备类型嵌入层
        self.device_type_embedding = nn.Embedding(
            num_embeddings=10,  # 支持最多10种设备类型
            embedding_dim=device_type_dim
        )
        
        # 设备类型特征提取
        self.device_type_mlp = nn.Sequential(
            nn.Linear(device_type_dim, device_type_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(device_type_dim * 2, device_type_dim),
            nn.Tanh()  # 使用Tanh激活函数，输出范围在[-1, 1]
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"DeviceTypeEncoder initialized with dim {device_type_dim}")
    
    def _init_weights(self):
        """初始化网络权重"""
        nn.init.normal_(self.device_type_embedding.weight, mean=0, std=0.1)
        for module in self.device_type_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, device_type_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            device_type_id: 设备类型ID [batch_size] 或标量
            
        Returns:
            torch.Tensor: 设备类型特征 [batch_size, device_type_dim]
        """
        # 确保输入是2D张量
        if device_type_id.dim() == 0:
            device_type_id = device_type_id.unsqueeze(0)
        
        # 设备类型嵌入
        device_embed = self.device_type_embedding(device_type_id)
        
        # 特征提取
        device_features = self.device_type_mlp(device_embed)
        
        return device_features


class DeviceAwareFiLMGenerator(nn.Module):
    """
    设备感知的FiLM参数生成器
    结合全局原型和设备类型信息生成FiLM参数
    """
    
    def __init__(
        self, 
        prototype_dim: int = 128,
        device_type_dim: int = 16,
        output_dim: int = 128,
        hidden_dims: list = [256, 512]
    ):
        """
        初始化设备感知FiLM生成器
        
        Args:
            prototype_dim: 全局原型维度
            device_type_dim: 设备类型特征维度
            output_dim: 输出维度 (FiLM参数维度)
            hidden_dims: 隐藏层维度列表
        """
        super(DeviceAwareFiLMGenerator, self).__init__()
        
        self.prototype_dim = prototype_dim
        self.device_type_dim = device_type_dim
        self.output_dim = output_dim
        
        # 设备类型编码器
        self.device_type_encoder = DeviceTypeEncoder(device_type_dim)
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(prototype_dim + device_type_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"DeviceAwareFiLMGenerator initialized: {prototype_dim + device_type_dim} -> {output_dim}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.fusion_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(
        self, 
        global_prototype: torch.Tensor, 
        device_type_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            global_prototype: 全局原型 [prototype_dim] 或 [batch_size, prototype_dim]
            device_type_id: 设备类型ID [batch_size] 或标量
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (gamma, beta) 调制参数
        """
        # 确保全局原型是2D张量
        if global_prototype.dim() == 1:
            global_prototype = global_prototype.unsqueeze(0)
        
        # 获取设备类型特征
        device_features = self.device_type_encoder(device_type_id)
        
        # 确保设备类型特征的batch维度与全局原型匹配
        if device_features.shape[0] != global_prototype.shape[0]:
            # 如果batch大小不匹配，重复设备类型特征
            device_features = device_features.repeat(global_prototype.shape[0], 1)
        
        # 融合全局原型和设备类型特征
        fused_features = torch.cat([global_prototype, device_features], dim=1)
        
        # 生成FiLM参数
        film_params = self.fusion_network(fused_features)
        
        # 分割为gamma和beta
        channels = self.output_dim // 2
        gamma = film_params[:, :channels]
        beta = film_params[:, channels:]
        
        return gamma, beta


class DeviceAwareAdapter(nn.Module):
    """
    设备感知的适配器网络
    结合设备类型信息进行条件调制
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_channels: int = 32,
        output_channels: int = 64,
        device_type_dim: int = 16,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        """
        初始化设备感知适配器
        
        Args:
            input_channels: 输入通道数
            hidden_channels: 隐藏通道数
            output_channels: 输出通道数
            device_type_dim: 设备类型特征维度
            kernel_size: 卷积核大小
            stride: 卷积步长
            padding: 填充大小
        """
        super(DeviceAwareAdapter, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.device_type_dim = device_type_dim
        
        # 设备类型编码器
        self.device_type_encoder = DeviceTypeEncoder(device_type_dim)
        
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
        
        # 设备类型调制层
        self.device_modulation = nn.Sequential(
            nn.Linear(device_type_dim, hidden_channels),
            nn.Sigmoid()  # 使用Sigmoid作为门控机制
        )
        
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
        
        logger.info(f"DeviceAwareAdapter initialized: {input_channels} -> {hidden_channels} -> {output_channels}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.01)
    
    def forward(
        self, 
        x: torch.Tensor, 
        film_params: Tuple[torch.Tensor, torch.Tensor],
        device_type_id: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图 [batch_size, input_channels, height, width]
            film_params: FiLM参数 (gamma, beta)
            device_type_id: 设备类型ID [batch_size] 或标量
            
        Returns:
            torch.Tensor: 调制后的特征图 [batch_size, output_channels, height, width]
        """
        gamma, beta = film_params
        
        # 第一个卷积
        u = self.conv1(x)
        
        # 层归一化
        batch_size, channels, height, width = u.shape
        u_flat = u.view(batch_size, channels, -1)
        u_norm = self.layer_norm(u_flat.transpose(1, 2)).transpose(1, 2)
        u_norm = u_norm.view(batch_size, channels, height, width)
        
        # 激活函数
        u_norm = self.activation(u_norm)
        
        # 设备类型调制
        # 确保设备类型ID是long类型
        if device_type_id.dtype != torch.long:
            device_type_id = device_type_id.long()
        
        # 获取设备类型特征
        device_features = self.device_type_encoder(device_type_id)
        device_gate = self.device_modulation(device_features)  # [batch_size, hidden_channels]
        device_gate = device_gate.unsqueeze(-1).unsqueeze(-1)  # [batch_size, hidden_channels, 1, 1]
        u_norm = u_norm * device_gate
        
        # FiLM调制
        gamma_expanded = gamma.unsqueeze(-1).unsqueeze(-1)
        beta_expanded = beta.unsqueeze(-1).unsqueeze(-1)
        u_modulated = u_norm * gamma_expanded + beta_expanded
        
        # 第二个卷积
        output = self.conv2(u_modulated)
        
        return output


class AdaptivePrototypeAlignment(nn.Module):
    """
    自适应原型对齐模块
    根据设备类型调整原型对齐的权重
    """
    
    def __init__(self, device_type_dim: int = 16, max_lambda: float = 1.0):
        """
        初始化自适应原型对齐模块
        
        Args:
            device_type_dim: 设备类型特征维度
            max_lambda: 最大lambda值
        """
        super(AdaptivePrototypeAlignment, self).__init__()
        
        self.device_type_dim = device_type_dim
        self.max_lambda = max_lambda
        
        # 设备类型编码器
        self.device_type_encoder = DeviceTypeEncoder(device_type_dim)
        
        # Lambda预测网络
        self.lambda_predictor = nn.Sequential(
            nn.Linear(device_type_dim, device_type_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(device_type_dim * 2, 1),
            nn.Sigmoid()  # 输出范围在[0, 1]
        )
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"AdaptivePrototypeAlignment initialized with max_lambda {max_lambda}")
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.lambda_predictor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, device_type_id: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            device_type_id: 设备类型ID [batch_size] 或标量
            
        Returns:
            torch.Tensor: 自适应lambda值 [batch_size, 1]
        """
        # 获取设备类型特征
        device_features = self.device_type_encoder(device_type_id)
        
        # 预测lambda值
        lambda_raw = self.lambda_predictor(device_features)
        
        # 缩放到[0, max_lambda]
        lambda_value = lambda_raw * self.max_lambda
        
        return lambda_value


# 设备类型映射
DEVICE_TYPE_MAPPING = {
    "id_00": 0,  # DUE Fan
    "id_02": 0,  # DUE Fan
    "id_04": 0,  # DUE Fan
    "id_06": 0,  # DUE Fan
    "dev_fan": 1,  # DEV Fan
    "dev_valve": 2,  # DEV Valve
}


def get_device_type_id(client_id: str) -> int:
    """
    根据客户端ID获取设备类型ID
    
    Args:
        client_id: 客户端ID
        
    Returns:
        int: 设备类型ID
    """
    return DEVICE_TYPE_MAPPING.get(client_id, 0)


if __name__ == "__main__":
    # 测试设备感知适配器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试参数
    batch_size = 4
    input_channels = 1
    hidden_channels = 32
    output_channels = 64
    height, width = 128, 32
    prototype_dim = 128
    device_type_dim = 16
    
    # 创建测试数据
    x = torch.randn(batch_size, input_channels, height, width)
    global_prototype = torch.randn(prototype_dim)
    device_type_id = torch.tensor([0, 1, 2, 0])  # 不同的设备类型
    
    # 测试设备类型编码器
    print("Testing DeviceTypeEncoder...")
    device_encoder = DeviceTypeEncoder(device_type_dim)
    device_features = device_encoder(device_type_id)
    print(f"Device features shape: {device_features.shape}")
    
    # 测试设备感知FiLM生成器
    print("\nTesting DeviceAwareFiLMGenerator...")
    film_generator = DeviceAwareFiLMGenerator(
        prototype_dim=prototype_dim,
        device_type_dim=device_type_dim,
        output_dim=hidden_channels * 2
    )
    
    gamma, beta = film_generator(global_prototype, device_type_id)
    print(f"Gamma shape: {gamma.shape}, Beta shape: {beta.shape}")
    
    # 测试设备感知适配器
    print("\nTesting DeviceAwareAdapter...")
    adapter = DeviceAwareAdapter(
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        device_type_dim=device_type_dim
    )
    
    output = adapter(x, (gamma, beta), device_type_id)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    # 测试自适应原型对齐
    print("\nTesting AdaptivePrototypeAlignment...")
    adaptive_alignment = AdaptivePrototypeAlignment(device_type_dim=device_type_dim)
    lambda_values = adaptive_alignment(device_type_id)
    print(f"Lambda values shape: {lambda_values.shape}")
    print(f"Lambda values: {lambda_values.squeeze()}")
    
    # 测试设备类型映射
    print("\nTesting device type mapping...")
    for client_id in ["id_00", "id_02", "dev_fan", "dev_valve"]:
        device_type = get_device_type_id(client_id)
        print(f"Client {client_id} -> Device type {device_type}")
    
    print("\nAll tests passed!")
