"""
Test Script for Fed-ProFiLA-AD
测试训练好的模型
"""

import os
import sys
import torch
import yaml
import logging
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from utils.seeding import set_seed
from dataset_loader.dataset_mimii import create_client_dataloader, get_client_ids
from models.backbone_cnn import create_backbone
from eval.inference import evaluate_all_clients
from eval.metrics import print_metrics

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        tuple: (模型状态, 全局原型, 训练历史)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_state = checkpoint['global_model_state']
    global_prototype = checkpoint['global_prototype']
    training_history = checkpoint.get('training_history', {})
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Model state keys: {list(model_state.keys())}")
    logger.info(f"Global prototype shape: {global_prototype.shape}")
    
    return model_state, global_prototype, training_history


def test_single_client(model, test_loader, global_prototype, local_prototype, device, client_id):
    """
    测试单个客户端
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        global_prototype: 全局原型
        local_prototype: 本地原型
        device: 设备
        client_id: 客户端ID
        
    Returns:
        dict: 评估结果
    """
    from eval.inference import evaluate_client
    
    return evaluate_client(
        model=model,
        test_loader=test_loader,
        global_prototype=global_prototype,
        local_prototype=local_prototype,
        device=device,
        client_id=client_id
    )


def test_all_clients(config, checkpoint_path, device):
    """
    测试所有客户端
    
    Args:
        config: 配置字典
        checkpoint_path: 检查点路径
        device: 设备
    """
    logger.info("Starting model testing...")
    
    # 加载检查点
    model_state, global_prototype, training_history = load_checkpoint(checkpoint_path, device)
    
    # 获取客户端ID
    client_ids = get_client_ids(config['dataset']['root_path'])
    logger.info(f"Found clients: {client_ids}")
    
    # 创建全局模型
    global_model = create_backbone(
        backbone_type="dcase",
        input_channels=config['model']['backbone']['input_channels'],
        feature_dim=config['model']['backbone']['feature_dim'],
        prototype_dim=config['model']['film_generator']['input_dim']
    )
    
    # 加载模型权重
    global_model.load_state_dict(model_state)
    global_model = global_model.to(device)
    global_model.eval()
    
    # 测试所有客户端
    eval_results = {}
    models = {}
    test_loaders = {}
    local_prototypes = {}
    
    logger.info("Creating test data loaders and computing local prototypes...")
    
    for client_id in client_ids:
        logger.info(f"Processing client {client_id}...")
        
        # 创建测试数据加载器
        test_loader = create_client_dataloader(
            root_path=config['dataset']['root_path'],
            client_id=client_id,
            is_train=False,
            batch_size=config['training']['batch_size'],
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory'],
            sample_rate=config['dataset']['sample_rate'],
            segment_length=config['dataset']['segment_length'],
            n_mels=config['dataset']['n_mels'],
            hop_length=config['dataset']['hop_length'],
            n_fft=config['dataset']['n_fft']
        )
        
        # 创建客户端模型副本
        client_model = create_backbone(
            backbone_type="dcase",
            input_channels=config['model']['backbone']['input_channels'],
            feature_dim=config['model']['backbone']['feature_dim'],
            prototype_dim=config['model']['film_generator']['input_dim']
        )
        client_model.load_state_dict(model_state)
        client_model = client_model.to(device)
        client_model.eval()
        
        # 计算本地原型
        from methods.fed_profila_ad import compute_local_prototype
        local_prototype = compute_local_prototype(
            model=client_model,
            dataloader=test_loader,
            global_prototype=global_prototype,
            device=device
        )
        
        models[client_id] = client_model
        test_loaders[client_id] = test_loader
        local_prototypes[client_id] = local_prototype
    
    # 运行评估
    logger.info("Running evaluation...")
    eval_results = evaluate_all_clients(
        models=models,
        test_loaders=test_loaders,
        global_prototype=global_prototype,
        local_prototypes=local_prototypes,
        device=device
    )
    
    return eval_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test Fed-ProFiLA-AD Model')
    parser.add_argument('--config', type=str, default='configs/mimii_due.yaml', help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='检查点路径')
    parser.add_argument('--device', type=str, default=None, help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 覆盖设备设置
    if args.device:
        config['system']['device'] = args.device
    
    # 设置设备
    if config['system']['device'] == 'cuda' and torch.cuda.is_available():
        gpu_id = config['system'].get('gpu_id', 0)
        if gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda:0')
        logger.info(f"Using CUDA device {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    logger.info("="*60)
    logger.info("Fed-ProFiLA-AD Model Testing")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    
    try:
        # 运行测试
        eval_results = test_all_clients(config, args.checkpoint, device)
        
        # 打印结果摘要
        logger.info("\n" + "="*50)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*50)
        
        if 'average' in eval_results:
            avg_metrics = eval_results['average']
            logger.info("Average Metrics:")
            for metric, value in avg_metrics.items():
                if not metric.endswith('_std'):
                    std_metric = f"{metric}_std"
                    std_value = avg_metrics.get(std_metric, 0.0)
                    logger.info(f"  {metric}: {value:.4f} ± {std_value:.4f}")
        
        # 保存测试结果
        import json
        os.makedirs('results', exist_ok=True)
        with open('results/test_results.json', 'w') as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        logger.info(f"\nTest results saved to 'results/test_results.json'")
        logger.info("Model testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise


if __name__ == "__main__":
    main()
