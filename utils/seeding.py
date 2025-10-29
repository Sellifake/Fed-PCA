"""
Seeding Utilities for Fed-ProFiLA-AD
实现随机种子设置，确保实验的可重现性
"""

import random
import numpy as np
import torch
import os
import logging

logger = logging.getLogger(__name__)


def set_seed(seed_value: int = 42) -> None:
    """
    设置所有随机种子以确保可重现性
    
    Args:
        seed_value: 种子值，默认为42
    """
    # Python内置random模块
    random.seed(seed_value)
    
    # NumPy随机种子
    np.random.seed(seed_value)
    
    # PyTorch随机种子
    torch.manual_seed(seed_value)
    
    # CUDA随机种子 (如果可用)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        
        # 设置CUDA确定性行为
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"CUDA random seed set to {seed_value}")
    
    # 设置环境变量以确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    logger.info(f"All random seeds set to {seed_value}")


def get_deterministic_worker_init_fn(seed: int = 42):
    """
    获取确定性数据加载器工作进程初始化函数
    
    Args:
        seed: 种子值
        
    Returns:
        callable: 工作进程初始化函数
    """
    def worker_init_fn(worker_id: int) -> None:
        """
        数据加载器工作进程初始化函数
        
        Args:
            worker_id: 工作进程ID
        """
        # 为每个工作进程设置不同的种子
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        # 如果使用CUDA，也设置CUDA种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(worker_seed)
            torch.cuda.manual_seed_all(worker_seed)
    
    return worker_init_fn


def create_deterministic_generator(seed: int = 42) -> torch.Generator:
    """
    创建确定性随机数生成器
    
    Args:
        seed: 种子值
        
    Returns:
        torch.Generator: 确定性随机数生成器
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def ensure_reproducibility(seed: int = 42) -> None:
    """
    确保完全的可重现性设置
    
    Args:
        seed: 种子值
    """
    # 设置所有随机种子
    set_seed(seed)
    
    # 设置PyTorch确定性行为
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    # 设置环境变量
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info(f"Reproducibility settings applied with seed {seed}")


def check_reproducibility() -> dict:
    """
    检查当前的可重现性设置
    
    Returns:
        dict: 可重现性设置状态
    """
    settings = {
        'torch_deterministic': torch.backends.cudnn.deterministic if hasattr(torch.backends, 'cudnn') else None,
        'torch_benchmark': torch.backends.cudnn.benchmark if hasattr(torch.backends, 'cudnn') else None,
        'python_hashseed': os.environ.get('PYTHONHASHSEED'),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return settings


def log_reproducibility_settings() -> None:
    """记录当前的可重现性设置"""
    settings = check_reproducibility()
    
    logger.info("Current reproducibility settings:")
    for key, value in settings.items():
        logger.info(f"  {key}: {value}")


def test_reproducibility(seed: int = 42, num_tests: int = 3) -> bool:
    """
    测试可重现性
    
    Args:
        seed: 种子值
        num_tests: 测试次数
        
    Returns:
        bool: 是否通过可重现性测试
    """
    logger.info(f"Testing reproducibility with seed {seed}...")
    
    results = []
    
    for i in range(num_tests):
        # 设置种子
        set_seed(seed)
        
        # 生成随机数
        python_random = random.random()
        numpy_random = np.random.random()
        torch_random = torch.rand(1).item()
        
        results.append((python_random, numpy_random, torch_random))
        
        logger.info(f"Test {i+1}: python={python_random:.6f}, numpy={numpy_random:.6f}, torch={torch_random:.6f}")
    
    # 检查所有结果是否相同
    all_same = all(
        results[0][0] == results[i][0] and
        results[0][1] == results[i][1] and
        results[0][2] == results[i][2]
        for i in range(1, num_tests)
    )
    
    if all_same:
        logger.info("✓ Reproducibility test PASSED")
    else:
        logger.warning("✗ Reproducibility test FAILED")
    
    return all_same


if __name__ == "__main__":
    # 测试种子设置功能
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试基本种子设置
    print("Testing basic seed setting...")
    set_seed(42)
    
    # 测试可重现性
    print("\nTesting reproducibility...")
    is_reproducible = test_reproducibility(42, 3)
    
    # 检查设置
    print("\nChecking reproducibility settings...")
    log_reproducibility_settings()
    
    # 测试确定性生成器
    print("\nTesting deterministic generator...")
    gen1 = create_deterministic_generator(42)
    gen2 = create_deterministic_generator(42)
    
    rand1 = torch.rand(5, generator=gen1)
    rand2 = torch.rand(5, generator=gen2)
    
    print(f"Generator 1: {rand1}")
    print(f"Generator 2: {rand2}")
    print(f"Generators produce same results: {torch.allclose(rand1, rand2)}")
    
    # 测试工作进程初始化函数
    print("\nTesting worker init function...")
    worker_init = get_deterministic_worker_init_fn(42)
    
    # 模拟工作进程初始化
    for worker_id in range(3):
        worker_init(worker_id)
        print(f"Worker {worker_id}: python={random.random():.6f}, numpy={np.random.random():.6f}, torch={torch.rand(1).item():.6f}")
    
    print(f"\nReproducibility test result: {'PASSED' if is_reproducible else 'FAILED'}")
