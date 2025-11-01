"""
Basic Fed-ProFiLA-AD Runner (基础版本)
实现基础的Fed-ProFiLA-AD训练脚本
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.seeding import set_seed
from dataset_loader.base_dataset import create_dataloader, get_client_ids
from models.backbone_cnn import create_backbone
from methods.fed_profila_ad import initialize_global_prototype
from trainers.server_loop import Server
from trainers.client_loop import Client, ClientManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fed_profila_ad.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Basic Fed-ProFiLA-AD Training')
    parser.add_argument('--config', type=str, default='configs/basic_federation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), default: auto-detect (prefers cuda)')
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Creating default config file...")
        # 创建默认配置
        default_config = {
            'dataset': {
                'root_path': 'data',
                'sample_rate': 16000,
                'segment_length': 4096,
                'n_mels': 128,
                'hop_length': 512,
                'n_fft': 1024,
                'max_train_samples': 500,  # 限制训练样本数（快速测试）
                'max_test_samples': 200,   # 限制测试样本数（快速测试）
                'include_abnormal_in_train': True,
                'abnormal_fraction': 0.2
            },
            'model': {
                'backbone_type': 'default',
                'input_channels': 1,
                'feature_dim': 128,
                'prototype_dim': 128,
                'dropout_rate': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.0001,
                'weight_decay': 0.0001,
                'optimizer': 'adam'
            },
            'federation': {
                'num_rounds': 20,  # 增加轮数
                'local_epochs': 2,  # 可根据需要调整
                'lambda_proto': 0.01,  # 降低到0.01，更稳定
                'client_selection': 'all'
            },
            'loss': {
                'lambda_contrastive': 0.5,
                'contrastive_margin': 0.8
            },
            'system': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',  # 默认GPU
                'seed': 42,
                'num_workers': 4,
                'pin_memory': True if torch.cuda.is_available() else False
            }
        }
        
        os.makedirs('configs', exist_ok=True)
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        logger.info(f"Default config saved to {args.config}")
        config = default_config
    else:
        config = load_config(args.config)
    
    # 设备选择逻辑：优先级 命令行参数 > GPU可用性检查 > 配置文件
    if args.device:
        # 命令行参数优先
        config['system']['device'] = args.device
        if args.device == 'cuda':
            config['system']['pin_memory'] = True
    else:
        # 如果没有命令行参数，强制检查GPU并优先使用
        if torch.cuda.is_available():
            config['system']['device'] = 'cuda'
            config['system']['pin_memory'] = True
            print(f"GPU available! Using CUDA (device count: {torch.cuda.device_count()})")
        else:
            config['system']['device'] = 'cpu'
            config['system']['pin_memory'] = False
            print("Warning: CUDA not available, using CPU")
    
    # 设置随机种子
    set_seed(config['system']['seed'])
    
    # 设置设备
    device_str = config['system']['device']
    # 如果配置是cuda但实际不可用，fallback到cpu
    if device_str == 'cuda' and not torch.cuda.is_available():
        print("Warning: Config specifies CUDA but it's not available, falling back to CPU")
        device_str = 'cpu'
        config['system']['device'] = 'cpu'
        config['system']['pin_memory'] = False
    
    device = torch.device(device_str)
    if device_str == 'cuda':
        print(f"Using device: {device} (GPU: {torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    run_dir = Path('outputs') / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取客户端ID列表
    root_path = config['dataset']['root_path']
    client_ids = get_client_ids(root_path)
    
    if not client_ids:
        logger.error(f"No clients found in {root_path}")
        return
    
    print(f"Found {len(client_ids)} clients: {client_ids}")
    
    # 创建全局模型
    model_config = config['model']
    backbone = create_backbone(
        backbone_type=model_config['backbone_type'],
        input_channels=model_config['input_channels'],
        feature_dim=model_config['feature_dim'],
        prototype_dim=model_config['prototype_dim'],
        dropout_rate=model_config['dropout_rate'],
        hidden_dims=model_config.get('hidden_dims', [64, 128, 256])
    ).to(device)
    
    feature_dim = backbone.get_feature_dim()
    prototype_dim = backbone.get_prototype_dim()
    
    # 初始化全局原型
    global_prototype = initialize_global_prototype(prototype_dim, device)
    
    # 创建服务器
    server = Server(
        global_model=backbone,
        device=device,
        client_selection_strategy=config['federation']['client_selection'],
        feature_dim=feature_dim,
        prototype_momentum=config['federation'].get('prototype_momentum', 0.7)
    )
    
    # 创建客户端管理器
    client_manager = ClientManager(device)
    
    # 为每个客户端创建数据加载器和客户端对象
    training_config = config['training']
    dataset_config = config['dataset']
    
    for client_id in client_ids:
        # 获取数据采样限制
        max_train_samples = dataset_config.get('max_train_samples', None)
        max_test_samples = dataset_config.get('max_test_samples', None)
        
        # 创建训练数据加载器
        train_loader = create_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=True,
            batch_size=training_config['batch_size'],
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory'],
            sample_rate=dataset_config['sample_rate'],
            segment_length=dataset_config['segment_length'],
            n_mels=dataset_config['n_mels'],
            hop_length=dataset_config['hop_length'],
            n_fft=dataset_config['n_fft'],
            max_samples=max_train_samples,
            include_abnormal_in_train=dataset_config.get('include_abnormal_in_train', True),
            abnormal_fraction=dataset_config.get('abnormal_fraction', 0.2)
        )
        
        # 创建测试数据加载器
        test_loader = create_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=False,
            batch_size=training_config['batch_size'],
            num_workers=config['system']['num_workers'],
            pin_memory=config['system']['pin_memory'],
            sample_rate=dataset_config['sample_rate'],
            segment_length=dataset_config['segment_length'],
            n_mels=dataset_config['n_mels'],
            hop_length=dataset_config['hop_length'],
            n_fft=dataset_config['n_fft'],
            max_samples=max_test_samples
        )
        
        if train_loader is None or test_loader is None:
            logger.warning(f"Client {client_id} has empty dataset, skipping...")
            continue
        
        if len(train_loader.dataset) == 0 or len(test_loader.dataset) == 0:
            logger.warning(f"Client {client_id} has no data, skipping...")
            continue
        
        # 创建客户端
        client = Client(
            client_id=client_id,
            model=backbone,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=training_config['learning_rate'],
            optimizer_name=training_config['optimizer'],
            weight_decay=training_config['weight_decay']
        )
        
        client_manager.add_client(client)
    
    if len(client_manager.clients) == 0:
        logger.error("No valid clients found!")
        return
    
    # 运行联邦学习
    print(f"\nStarting federated learning with {len(client_manager.clients)} clients...")
    print(f"Training samples per client: ~{max_train_samples or 'all'}, Test samples: ~{max_test_samples or 'all'}")
    
    num_rounds = config['federation']['num_rounds']
    local_epochs = config['federation']['local_epochs']
    lambda_proto = config['federation'].get('lambda_proto', 0.01)  # 默认0.01
    lambda_contrastive = config.get('loss', {}).get('lambda_contrastive', 0.5)
    contrastive_margin = config.get('loss', {}).get('contrastive_margin', 0.8)
    
    # 保存训练历史
    training_history = {
        'rounds': [],
        'eval_metrics': []
    }
    
    best_avg_auc = -1.0
    best_checkpoint = None
    for round_idx in range(num_rounds):
        print(f"\n[Round {round_idx + 1}/{num_rounds}]")
        
        # 获取全局模型状态和原型
        global_theta = server.get_global_model_state()
        global_mu = server.get_global_prototype()
        
        # 选择客户端
        available_clients = list(client_manager.get_all_clients().keys())
        selected_clients = server.select_clients(available_clients)
        
        # 训练客户端（简化日志输出）
        print(f"Training {len(selected_clients)} clients...")
        client_results, client_prototypes = client_manager.train_clients(
            global_theta=global_theta,
            global_mu=global_mu,
            selected_clients=selected_clients,
            local_epochs=local_epochs,
            lambda_proto=lambda_proto,
            lambda_contrastive=lambda_contrastive,
            contrastive_margin=contrastive_margin,
            lambda_separation=config.get('loss', {}).get('lambda_separation', 0.5),
            separation_margin=config.get('loss', {}).get('separation_margin', 0.8),
            lambda_supcon=config.get('loss', {}).get('lambda_supcon', 1.0),
            temperature=config.get('loss', {}).get('temperature', 0.2)
        )
        
        # 获取客户端权重
        client_weights = client_manager.get_client_weights()
        selected_weights = {cid: client_weights[cid] for cid in selected_clients}
        
        # 服务器聚合
        server.aggregate_models(client_results, selected_weights)
        server.aggregate_prototypes(client_prototypes, selected_weights)
        
        # 每轮评估（简化输出）
        eval_results = client_manager.evaluate_clients(
            global_theta=server.get_global_model_state(),
            global_mu=server.get_global_prototype()
        )
        
        # 计算平均指标
        all_aucs = [metrics.get('auc', 0.0) for metrics in eval_results.values()]
        all_f1s = [metrics.get('f1', 0.0) for metrics in eval_results.values()]
        all_precisions = [metrics.get('precision', 0.0) for metrics in eval_results.values()]
        all_recalls = [metrics.get('recall', 0.0) for metrics in eval_results.values()]
        if all_aucs:
            avg_auc = np.mean(all_aucs)
            avg_f1 = np.mean(all_f1s) if all_f1s else 0.0
            avg_precision = np.mean(all_precisions) if all_precisions else 0.0
            avg_recall = np.mean(all_recalls) if all_recalls else 0.0
            # 显示每个客户端的loss（从训练历史中获取）以及分量
            avg_loss = np.mean([
                client.training_history['losses'][-1] if client.training_history['losses'] else 0.0
                for client_id, client in client_manager.clients.items()
                if client_id in selected_clients
            ])
            avg_task_loss = np.mean([
                client.training_history['task_losses'][-1] if client.training_history['task_losses'] else 0.0
                for client_id, client in client_manager.clients.items()
                if client_id in selected_clients
            ])
            avg_proto_loss = np.mean([
                client.training_history['proto_losses'][-1] if client.training_history['proto_losses'] else 0.0
                for client_id, client in client_manager.clients.items()
                if client_id in selected_clients
            ])
            
            print(f"  Loss: {avg_loss:.4f} (task {avg_task_loss:.4f}, proto {avg_proto_loss:.4f}) | "
                  f"AUC: {avg_auc:.4f}, F1: {avg_f1:.4f}, P: {avg_precision:.4f}, R: {avg_recall:.4f}")
            
            # 保存评估结果（完整保存，但只显示AUC）
            training_history['rounds'].append(round_idx + 1)
            training_history['eval_metrics'].append({
                'round': round_idx + 1,
                'avg_auc': float(avg_auc),
                'avg_f1': float(avg_f1),
                'avg_precision': float(avg_precision),
                'avg_recall': float(avg_recall),
                'avg_loss': float(avg_loss),
                'avg_task_loss': float(avg_task_loss),
                'avg_proto_loss': float(avg_proto_loss),
                'client_metrics': {k: {
                    'auc': float(v.get('auc', 0.0)),
                    'f1': float(v.get('f1', 0.0)),
                    'precision': float(v.get('precision', 0.0)),
                    'recall': float(v.get('recall', 0.0))
                } for k, v in eval_results.items()}
            })

            # 保存最佳模型（按平均AUC）
            if avg_auc > best_avg_auc:
                best_avg_auc = avg_auc
                best_checkpoint = {
                    'model_state': server.get_global_model_state(),
                    'global_prototype': server.get_global_prototype().cpu(),
                    'config': config,
                    'training_history': training_history,
                    'best_round': round_idx + 1,
                    'best_avg_auc': float(best_avg_auc)
                }
                best_path = run_dir / "best_model.pt"
                torch.save(best_checkpoint, best_path)
                print(f"  [Checkpoint] New best AUC {best_avg_auc:.4f} at round {round_idx + 1}, saved to: {best_path}")
    
    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)
    
    # 保存最终模型
    checkpoint = {
        'model_state': server.get_global_model_state(),
        'global_prototype': server.get_global_prototype().cpu(),
        'config': config,
        'training_history': training_history
    }
    
    checkpoint_path = run_dir / "final_model.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # 保存训练历史为JSON
    history_path = run_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # 保存配置
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 打印最终总结（简化）
    if training_history['eval_metrics']:
        final_metrics = training_history['eval_metrics'][-1]
        print(f"\nFinal Results:")
        print(f"  Average AUC: {final_metrics['avg_auc']:.4f}")
        print(f"  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
