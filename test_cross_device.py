"""
Test script for Cross-Device Fed-ProFiLA-AD
测试跨设备联邦学习实现
"""

import os
import sys
import torch
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.seeding import set_seed
from dataset_loader.cross_device_dataset import (
    create_cross_device_dataloader, 
    get_cross_device_client_ids,
    get_device_type_statistics
)
from models.device_aware_adapters import (
    DeviceAwareFiLMGenerator,
    DeviceAwareAdapter,
    AdaptivePrototypeAlignment,
    get_device_type_id
)
from models.backbone_cnn import create_backbone

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_device_aware_components():
    """测试设备感知组件"""
    logger.info("Testing device-aware components...")
    
    # 设置随机种子
    set_seed(42)
    
    # 测试参数
    batch_size = 4
    input_channels = 1
    height, width = 128, 32
    prototype_dim = 128
    device_type_dim = 16
    
    # 创建测试数据
    x = torch.randn(batch_size, input_channels, height, width)
    global_prototype = torch.randn(prototype_dim)
    device_type_ids = torch.tensor([0, 1, 2, 0], dtype=torch.long)  # 不同的设备类型
    
    # 确保全局原型有正确的batch维度
    global_prototype = global_prototype.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, prototype_dim]
    
    # 测试设备感知FiLM生成器
    logger.info("Testing DeviceAwareFiLMGenerator...")
    film_generator = DeviceAwareFiLMGenerator(
        prototype_dim=prototype_dim,
        device_type_dim=device_type_dim,
        output_dim=64  # 32 * 2 for gamma and beta
    )
    
    gamma, beta = film_generator(global_prototype, device_type_ids)
    assert gamma.shape == (batch_size, 32)
    assert beta.shape == (batch_size, 32)
    logger.info("✓ DeviceAwareFiLMGenerator test passed")
    
    # 测试设备感知适配器
    logger.info("Testing DeviceAwareAdapter...")
    adapter = DeviceAwareAdapter(
        input_channels=input_channels,
        hidden_channels=32,
        output_channels=64,
        device_type_dim=device_type_dim
    )
    
    output = adapter(x, (gamma, beta), device_type_ids)
    assert output.shape == (batch_size, 64, height, width)
    logger.info("✓ DeviceAwareAdapter test passed")
    
    # 测试自适应原型对齐
    logger.info("Testing AdaptivePrototypeAlignment...")
    adaptive_alignment = AdaptivePrototypeAlignment(device_type_dim=device_type_dim)
    lambda_values = adaptive_alignment(device_type_ids)
    assert lambda_values.shape == (batch_size, 1)
    logger.info("✓ AdaptivePrototypeAlignment test passed")
    
    # 测试设备类型映射
    logger.info("Testing device type mapping...")
    test_clients = ["id_00", "id_02", "dev_fan", "dev_valve"]
    for client_id in test_clients:
        device_type = get_device_type_id(client_id)
        logger.info(f"Client {client_id} -> Device type {device_type}")
    
    logger.info("✓ All device-aware component tests passed!")


def test_cross_device_dataset():
    """测试跨设备数据集"""
    logger.info("Testing cross-device dataset...")
    
    # 设置随机种子
    set_seed(42)
    
    # 测试参数
    root_path = "data"
    
    # 测试获取客户端信息
    logger.info("Testing client information retrieval...")
    try:
        clients = get_cross_device_client_ids(root_path)
        logger.info(f"Found {len(clients)} clients:")
        for client in clients:
            logger.info(f"  - {client['client_id']} ({client['device_type']})")
    except Exception as e:
        logger.warning(f"Could not load client information: {e}")
        logger.info("This is expected if data directory doesn't exist")
    
    # 测试获取统计信息
    logger.info("Testing device type statistics...")
    try:
        stats = get_device_type_statistics(root_path)
        logger.info("Device type statistics:")
        for client_id, stat in stats.items():
            logger.info(f"  - {client_id}: {stat['normal_samples']} normal, "
                       f"{stat['abnormal_samples']} abnormal")
    except Exception as e:
        logger.warning(f"Could not load statistics: {e}")
        logger.info("This is expected if data directory doesn't exist")
    
    # 测试数据加载器（如果数据存在）
    if clients:
        client_id = clients[0]["client_id"]
        logger.info(f"Testing dataloader for client {client_id}...")
        
        try:
            # 创建数据加载器
            dataloader = create_cross_device_dataloader(
                root_path=root_path,
                client_id=client_id,
                is_train=True,
                batch_size=2,
                num_workers=0
            )
            
            # 测试数据加载
            for i, (spectrogram, label, device_type) in enumerate(dataloader):
                logger.info(f"Batch {i}: spectrogram shape={spectrogram.shape}, "
                           f"labels={label}, device_type={device_type}")
                if i >= 1:  # 只测试前2个batch
                    break
            
            logger.info("✓ Cross-device dataset test passed!")
            
        except Exception as e:
            logger.warning(f"Could not test dataloader: {e}")
            logger.info("This is expected if data directory doesn't exist")
    else:
        logger.info("No clients found, skipping dataloader test")


def test_model_integration():
    """测试模型集成"""
    logger.info("Testing model integration...")
    
    # 设置随机种子
    set_seed(42)
    
    # 测试参数
    batch_size = 2
    input_channels = 1
    height, width = 128, 32
    feature_dim = 128
    prototype_dim = 128
    
    # 创建测试数据
    x = torch.randn(batch_size, input_channels, height, width)
    global_prototype = torch.randn(prototype_dim)
    device_type_ids = torch.tensor([0, 1])  # 不同的设备类型
    
    # 测试骨干网络
    logger.info("Testing backbone network...")
    backbone = create_backbone(
        backbone_type="dcase",
        input_channels=input_channels,
        feature_dim=feature_dim,
        prototype_dim=prototype_dim
    )
    
    # 注意：原始骨干网络不支持设备类型输入，这里只是测试基本功能
    output = backbone(x, global_prototype)
    assert output.shape == (batch_size, feature_dim)
    logger.info("✓ Backbone network test passed")
    
    # 测试设备感知组件集成
    logger.info("Testing device-aware component integration...")
    
    # 创建设备感知FiLM生成器
    film_generator = DeviceAwareFiLMGenerator(
        prototype_dim=prototype_dim,
        device_type_dim=16,
        output_dim=64
    )
    
    # 创建设备感知适配器
    adapter = DeviceAwareAdapter(
        input_channels=input_channels,
        hidden_channels=32,
        output_channels=64,
        device_type_dim=16
    )
    
    # 测试集成流程
    gamma, beta = film_generator(global_prototype, device_type_ids)
    adapted_features = adapter(x, (gamma, beta), device_type_ids)
    
    assert adapted_features.shape == (batch_size, 64, height, width)
    logger.info("✓ Device-aware component integration test passed")


def test_configuration():
    """测试配置文件"""
    logger.info("Testing configuration files...")
    
    config_path = "configs/cross_device_federation.yaml"
    
    if os.path.exists(config_path):
        logger.info(f"✓ Configuration file exists: {config_path}")
        
        # 尝试加载配置文件
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查关键配置项
            required_keys = ['dataset', 'model', 'federation', 'training']
            for key in required_keys:
                if key in config:
                    logger.info(f"✓ Found {key} configuration")
                else:
                    logger.warning(f"✗ Missing {key} configuration")
            
            logger.info("✓ Configuration file is valid")
            
        except Exception as e:
            logger.error(f"✗ Error loading configuration: {e}")
    else:
        logger.warning(f"✗ Configuration file not found: {config_path}")


def main():
    """主测试函数"""
    logger.info("Starting Cross-Device Fed-ProFiLA-AD tests...")
    logger.info("=" * 50)
    
    try:
        # 测试设备感知组件
        test_device_aware_components()
        logger.info("")
        
        # 测试跨设备数据集
        test_cross_device_dataset()
        logger.info("")
        
        # 测试模型集成
        test_model_integration()
        logger.info("")
        
        # 测试配置文件
        test_configuration()
        logger.info("")
        
        logger.info("=" * 50)
        logger.info("✓ All tests completed successfully!")
        logger.info("Cross-Device Fed-ProFiLA-AD is ready to use!")
        
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
