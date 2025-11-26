"""
Advanced Fed-ProFiLA-AD Runner (创新版本)
基于run_basic.py的三个创新点：
1. 自适应客户端重要性加权：根据数据量和性能动态调整聚合权重
2. 渐进式原型对齐：原型对齐权重随训练逐渐增加，提升收敛稳定性
3. 客户端性能感知的学习率调度：根据客户端AUC性能动态调整学习率
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

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
        logging.FileHandler('logs/advanced_fed_profila_ad.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdaptiveClientWeightManager:
    """
    创新点1: 自适应客户端重要性加权管理器
    根据数据量和性能动态调整聚合权重
    """
    
    def __init__(self, base_weight_alpha: float = 0.5, performance_beta: float = 0.5):
        """
        Args:
            base_weight_alpha: 基于数据量的权重比例 (0-1)
            performance_beta: 基于性能的权重比例 (0-1)，总和应为1
        """
        self.base_weight_alpha = base_weight_alpha
        self.performance_beta = performance_beta
        self.client_performance_history = {}  # 记录客户端历史性能
        
    def compute_adaptive_weights(
        self,
        client_ids: List[str],
        client_data_sizes: Dict[str, int],
        client_performances: Dict[str, float]
    ) -> Dict[str, float]:
        """
        计算自适应权重
        
        Args:
            client_ids: 客户端ID列表
            client_data_sizes: 客户端数据量字典
            client_performances: 客户端性能字典 (AUC)
            
        Returns:
            归一化的权重字典
        """
        # 更新性能历史
        for client_id in client_ids:
            if client_id not in self.client_performance_history:
                self.client_performance_history[client_id] = []
            if client_id in client_performances:
                self.client_performance_history[client_id].append(client_performances[client_id])
                # 只保留最近5次性能记录
                if len(self.client_performance_history[client_id]) > 5:
                    self.client_performance_history[client_id].pop(0)
        
        weights = {}
        
        # 计算数据量权重（归一化）
        if client_data_sizes:
            total_size = sum(client_data_sizes.values())
            if total_size > 0:
                data_weights = {
                    cid: client_data_sizes[cid] / total_size 
                    for cid in client_ids if cid in client_data_sizes
                }
            else:
                data_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        else:
            data_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        
        # 计算性能权重（基于平均性能）
        if client_performances:
            # 使用历史平均性能
            avg_performances = {}
            for client_id in client_ids:
                if client_id in self.client_performance_history and self.client_performance_history[client_id]:
                    avg_performances[client_id] = np.mean(self.client_performance_history[client_id])
                elif client_id in client_performances:
                    avg_performances[client_id] = client_performances[client_id]
                else:
                    avg_performances[client_id] = 0.5  # 默认值
            
            # 归一化性能权重（使用softmax确保稳定性）
            if avg_performances:
                perf_values = np.array([avg_performances.get(cid, 0.5) for cid in client_ids])
                # 使用温度缩放的softmax
                temperature = 2.0
                exp_perf = np.exp(perf_values / temperature)
                perf_weights = {
                    cid: exp_perf[i] / exp_perf.sum()
                    for i, cid in enumerate(client_ids)
                }
            else:
                perf_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        else:
            perf_weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        
        # 组合权重
        for client_id in client_ids:
            weights[client_id] = (
                self.base_weight_alpha * data_weights.get(client_id, 0.0) +
                self.performance_beta * perf_weights.get(client_id, 0.0)
            )
        
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {cid: w / total_weight for cid, w in weights.items()}
        else:
            weights = {cid: 1.0 / len(client_ids) for cid in client_ids}
        
        return weights


class ProgressivePrototypeAlignment:
    """
    创新点2: 渐进式原型对齐
    原型对齐权重随训练逐渐增加，提升收敛稳定性
    """
    
    def __init__(self, initial_lambda: float = 0.001, final_lambda: float = 0.01, warmup_rounds: int = 5):
        """
        Args:
            initial_lambda: 初始原型对齐权重
            final_lambda: 最终原型对齐权重
            warmup_rounds: 预热轮数
        """
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_rounds = warmup_rounds
        
    def get_lambda(self, current_round: int, total_rounds: int) -> float:
        """
        获取当前轮次的原型对齐权重
        
        Args:
            current_round: 当前轮次（从0开始）
            total_rounds: 总轮数
            
        Returns:
            当前的原型对齐权重
        """
        if current_round < self.warmup_rounds:
            # 预热阶段：线性增加
            progress = current_round / self.warmup_rounds
            return self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        else:
            # 主训练阶段：使用余弦退火策略逐渐增加到最终值
            warmup_progress = (current_round - self.warmup_rounds) / max(1, total_rounds - self.warmup_rounds)
            # 余弦函数：从final_lambda的0.8倍逐渐增加到final_lambda
            cosine_factor = 0.8 + 0.2 * (1 - np.cos(np.pi * warmup_progress / 2))
            return self.final_lambda * cosine_factor


class PerformanceAwareLearningRateScheduler:
    """
    创新点3: 客户端性能感知的学习率调度器
    根据客户端AUC性能动态调整学习率
    """
    
    def __init__(
        self,
        base_lr: float = 0.0001,
        min_lr: float = 0.00001,
        max_lr: float = 0.001,
        performance_threshold: float = 0.7,
        lr_adjust_factor: float = 1.2
    ):
        """
        Args:
            base_lr: 基础学习率
            min_lr: 最小学习率
            max_lr: 最大学习率
            performance_threshold: 性能阈值（低于此值增加学习率）
            lr_adjust_factor: 学习率调整因子
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.performance_threshold = performance_threshold
        self.lr_adjust_factor = lr_adjust_factor
        self.client_lr_history = {}  # 记录每个客户端的学习率历史
        
    def get_adaptive_lr(self, client_id: str, current_performance: float, current_round: int) -> float:
        """
        根据客户端性能获取自适应学习率
        
        Args:
            client_id: 客户端ID
            current_performance: 当前AUC性能
            current_round: 当前轮次
            
        Returns:
            调整后的学习率
        """
        if client_id not in self.client_lr_history:
            self.client_lr_history[client_id] = {
                'lr': self.base_lr,
                'best_performance': current_performance
            }
        
        history = self.client_lr_history[client_id]
        current_lr = history['lr']
        
        # 如果性能低于阈值，增加学习率（更积极地学习）
        if current_performance < self.performance_threshold:
            new_lr = min(self.max_lr, current_lr * self.lr_adjust_factor)
        # 如果性能提升，保持或略微降低学习率（稳定学习）
        elif current_performance > history['best_performance']:
            new_lr = max(self.min_lr, current_lr * 0.99)
            history['best_performance'] = current_performance
        # 性能稳定，保持当前学习率
        else:
            new_lr = current_lr
        
        # 更新历史
        history['lr'] = new_lr
        
        return new_lr


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Advanced Fed-ProFiLA-AD Training')
    parser.add_argument('--config', type=str, default='configs/basic_federation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), default: auto-detect (prefers cuda)')
    
    args = parser.parse_args()
    
    # 加载配置
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Creating default config file...")
        default_config = {
            'dataset': {
                'root_path': 'data',
                'sample_rate': 16000,
                'segment_length': 4096,
                'n_mels': 128,
                'hop_length': 512,
                'n_fft': 1024,
                'max_train_samples': 500,
                'max_test_samples': 200,
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
                'num_rounds': 20,
                'local_epochs': 2,
                'lambda_proto': 0.01,
                'client_selection': 'all'
            },
            'loss': {
                'lambda_contrastive': 0.5,
                'contrastive_margin': 0.8
            },
            'advanced': {
                'adaptive_weighting': {
                    'enabled': True,
                    'data_weight_alpha': 0.5,
                    'performance_weight_beta': 0.5
                },
                'progressive_prototype': {
                    'enabled': True,
                    'initial_lambda': 0.001,
                    'final_lambda': 0.01,
                    'warmup_rounds': 5
                },
                'adaptive_lr': {
                    'enabled': True,
                    'min_lr': 0.00001,
                    'max_lr': 0.001,
                    'performance_threshold': 0.7,
                    'lr_adjust_factor': 1.2
                }
            },
            'system': {
                'device': 'cuda' if torch.cuda.is_available() else 'cpu',
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
    
    # 设备选择逻辑
    if args.device:
        config['system']['device'] = args.device
        if args.device == 'cuda':
            config['system']['pin_memory'] = True
    else:
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
    run_dir = Path('outputs') / f"advanced_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
    
    # 初始化创新组件
    advanced_config = config.get('advanced', {})
    
    # 获取各创新点的配置（如果不存在则使用默认值）
    adaptive_weighting_config = advanced_config.get('adaptive_weighting', {})
    progressive_prototype_config = advanced_config.get('progressive_prototype', {})
    adaptive_lr_config = advanced_config.get('adaptive_lr', {})
    
    # 创新点1: 自适应客户端权重管理器
    adaptive_weight_manager = None
    if adaptive_weighting_config.get('enabled', True):
        adaptive_weight_manager = AdaptiveClientWeightManager(
            base_weight_alpha=adaptive_weighting_config.get('data_weight_alpha', 0.5),
            performance_beta=adaptive_weighting_config.get('performance_weight_beta', 0.5)
        )
        logger.info("✓ Adaptive client weighting enabled")
    
    # 创新点2: 渐进式原型对齐
    progressive_proto = None
    if progressive_prototype_config.get('enabled', True):
        progressive_proto = ProgressivePrototypeAlignment(
            initial_lambda=progressive_prototype_config.get('initial_lambda', 0.001),
            final_lambda=progressive_prototype_config.get('final_lambda', 0.01),
            warmup_rounds=progressive_prototype_config.get('warmup_rounds', 5)
        )
        logger.info("✓ Progressive prototype alignment enabled")
    
    # 创新点3: 自适应学习率调度器
    adaptive_lr_scheduler = None
    if adaptive_lr_config.get('enabled', True):
        base_lr = config['training']['learning_rate']
        adaptive_lr_scheduler = PerformanceAwareLearningRateScheduler(
            base_lr=base_lr,
            min_lr=adaptive_lr_config.get('min_lr', 0.00001),
            max_lr=adaptive_lr_config.get('max_lr', 0.001),
            performance_threshold=adaptive_lr_config.get('performance_threshold', 0.7),
            lr_adjust_factor=adaptive_lr_config.get('lr_adjust_factor', 1.2)
        )
        logger.info("✓ Adaptive learning rate scheduling enabled")
    
    # 为每个客户端创建数据加载器和客户端对象
    training_config = config['training']
    dataset_config = config['dataset']
    
    client_data_sizes = {}  # 用于自适应权重计算
    
    for client_id in client_ids:
        max_train_samples = dataset_config.get('max_train_samples', None)
        max_test_samples = dataset_config.get('max_test_samples', None)
        
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
        
        # 记录数据量（用于自适应权重）
        client_data_sizes[client_id] = len(train_loader.dataset)
        
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
    print(f"\nStarting advanced federated learning with {len(client_manager.clients)} clients...")
    print("=" * 60)
    print("创新功能:")
    print("  1. 自适应客户端重要性加权")
    print("  2. 渐进式原型对齐")
    print("  3. 客户端性能感知的学习率调度")
    print("=" * 60)
    
    num_rounds = config['federation']['num_rounds']
    local_epochs = config['federation']['local_epochs']
    base_lambda_proto = config['federation'].get('lambda_proto', 0.01)
    lambda_contrastive = config.get('loss', {}).get('lambda_contrastive', 0.5)
    contrastive_margin = config.get('loss', {}).get('contrastive_margin', 0.8)
    
    # 保存训练历史
    training_history = {
        'rounds': [],
        'eval_metrics': [],
        'adaptive_weights': [],
        'progressive_lambdas': [],
        'adaptive_lrs': []
    }
    
    best_avg_auc = -1.0
    best_checkpoint = None
    
    for round_idx in range(num_rounds):
        print(f"\n[Round {round_idx + 1}/{num_rounds}]")
        
        # 创新点2: 获取渐进式原型对齐权重
        if progressive_proto:
            lambda_proto = progressive_proto.get_lambda(round_idx, num_rounds)
            training_history['progressive_lambdas'].append({
                'round': round_idx + 1,
                'lambda_proto': float(lambda_proto)
            })
        else:
            lambda_proto = base_lambda_proto
        
        # 获取全局模型状态和原型
        global_theta = server.get_global_model_state()
        global_mu = server.get_global_prototype()
        
        # 选择客户端
        available_clients = list(client_manager.get_all_clients().keys())
        selected_clients = server.select_clients(available_clients)
        
        # 创新点3: 更新客户端学习率（如果需要，使用上一轮的性能）
        if adaptive_lr_scheduler and round_idx > 0:
            # 使用上一轮的评估结果
            if training_history['eval_metrics']:
                prev_metrics = training_history['eval_metrics'][-1]
                prev_client_metrics = prev_metrics.get('client_metrics', {})
                for client_id in selected_clients:
                    if client_id in prev_client_metrics:
                        current_perf = prev_client_metrics[client_id].get('auc', 0.5)
                        new_lr = adaptive_lr_scheduler.get_adaptive_lr(client_id, current_perf, round_idx)
                        # 更新客户端优化器的学习率
                        if client_id in client_manager.clients:
                            for param_group in client_manager.clients[client_id].optimizer.param_groups:
                                param_group['lr'] = new_lr
                            training_history['adaptive_lrs'].append({
                                'round': round_idx + 1,
                                'client_id': client_id,
                                'learning_rate': float(new_lr),
                                'performance': float(current_perf)
                            })
        
        # 训练客户端
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
        
        # 评估获取性能
        eval_results = client_manager.evaluate_clients(
            global_theta=server.get_global_model_state(),
            global_mu=server.get_global_prototype()
        )
        
        # 创新点1: 计算自适应权重
        if adaptive_weight_manager:
            client_performances = {cid: results.get('auc', 0.5) for cid, results in eval_results.items()}
            selected_data_sizes = {cid: client_data_sizes[cid] for cid in selected_clients if cid in client_data_sizes}
            adaptive_weights = adaptive_weight_manager.compute_adaptive_weights(
                selected_clients,
                selected_data_sizes,
                client_performances
            )
            training_history['adaptive_weights'].append({
                'round': round_idx + 1,
                'weights': {k: float(v) for k, v in adaptive_weights.items()}
            })
            selected_weights = adaptive_weights
        else:
            # 使用默认权重
            client_weights = client_manager.get_client_weights()
            selected_weights = {cid: client_weights[cid] for cid in selected_clients}
        
        # 服务器聚合
        server.aggregate_models(client_results, selected_weights)
        server.aggregate_prototypes(client_prototypes, selected_weights)
        
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
            
            print(f"  Lambda_proto: {lambda_proto:.6f} | "
                  f"Loss: {avg_loss:.4f} (task {avg_task_loss:.4f}, proto {avg_proto_loss:.4f}) | "
                  f"AUC: {avg_auc:.4f}, F1: {avg_f1:.4f}, P: {avg_precision:.4f}, R: {avg_recall:.4f}")
            
            # 保存评估结果
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
                'lambda_proto': float(lambda_proto),
                'client_metrics': {k: {
                    'auc': float(v.get('auc', 0.0)),
                    'f1': float(v.get('f1', 0.0)),
                    'precision': float(v.get('precision', 0.0)),
                    'recall': float(v.get('recall', 0.0))
                } for k, v in eval_results.items()}
            })
            
            # 保存最佳模型
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
            
            # 每轮保存CSV（实时更新）
            csv_data = []
            for metric in training_history['eval_metrics']:
                row = {
                    'round': metric['round'],
                    'avg_auc': metric['avg_auc'],
                    'avg_f1': metric['avg_f1'],
                    'avg_precision': metric['avg_precision'],
                    'avg_recall': metric['avg_recall'],
                    'avg_loss': metric['avg_loss'],
                    'avg_task_loss': metric['avg_task_loss'],
                    'avg_proto_loss': metric['avg_proto_loss'],
                    'lambda_proto': metric.get('lambda_proto', base_lambda_proto)
                }
                # 添加每个客户端的指标
                for client_id, client_metrics in metric['client_metrics'].items():
                    row[f'{client_id}_auc'] = client_metrics['auc']
                    row[f'{client_id}_f1'] = client_metrics['f1']
                    row[f'{client_id}_precision'] = client_metrics['precision']
                    row[f'{client_id}_recall'] = client_metrics['recall']
                csv_data.append(row)
            
            csv_path = run_dir / "training_history.csv"
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_path, index=False)
    
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
    
    # 保存训练历史为CSV
    csv_data = []
    for metric in training_history['eval_metrics']:
        row = {
            'round': metric['round'],
            'avg_auc': metric['avg_auc'],
            'avg_f1': metric['avg_f1'],
            'avg_precision': metric['avg_precision'],
            'avg_recall': metric['avg_recall'],
            'avg_loss': metric['avg_loss'],
            'avg_task_loss': metric['avg_task_loss'],
            'avg_proto_loss': metric['avg_proto_loss'],
            'lambda_proto': metric.get('lambda_proto', base_lambda_proto)
        }
        # 添加每个客户端的指标
        for client_id, client_metrics in metric['client_metrics'].items():
            row[f'{client_id}_auc'] = client_metrics['auc']
            row[f'{client_id}_f1'] = client_metrics['f1']
            row[f'{client_id}_precision'] = client_metrics['precision']
            row[f'{client_id}_recall'] = client_metrics['recall']
        csv_data.append(row)
    
    csv_path = run_dir / "training_history.csv"
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Training history CSV saved to: {csv_path}")
    
    # 绘制loss曲线图
    if training_history['eval_metrics']:
        plt.figure(figsize=(15, 10))
        
        rounds = [m['round'] for m in training_history['eval_metrics']]
        
        # 绘制loss曲线
        plt.subplot(2, 3, 1)
        avg_losses = [m['avg_loss'] for m in training_history['eval_metrics']]
        avg_task_losses = [m['avg_task_loss'] for m in training_history['eval_metrics']]
        avg_proto_losses = [m['avg_proto_loss'] for m in training_history['eval_metrics']]
        
        plt.plot(rounds, avg_losses, 'b-', label='Total Loss', linewidth=2, marker='o', markersize=4)
        plt.plot(rounds, avg_task_losses, 'r--', label='Task Loss', linewidth=2, marker='s', markersize=4)
        plt.plot(rounds, avg_proto_losses, 'g--', label='Proto Loss', linewidth=2, marker='^', markersize=4)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 绘制AUC曲线
        plt.subplot(2, 3, 2)
        avg_aucs = [m['avg_auc'] for m in training_history['eval_metrics']]
        plt.plot(rounds, avg_aucs, 'b-', label='Average AUC', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('AUC', fontsize=12)
        plt.title('Average AUC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 绘制F1曲线
        plt.subplot(2, 3, 3)
        avg_f1s = [m['avg_f1'] for m in training_history['eval_metrics']]
        plt.plot(rounds, avg_f1s, 'g-', label='Average F1', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Round', fontsize=12)
        plt.ylabel('F1 Score', fontsize=12)
        plt.title('Average F1 Score Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 绘制渐进式lambda_proto
        if training_history['progressive_lambdas']:
            plt.subplot(2, 3, 4)
            lambda_rounds = [m['round'] for m in training_history['progressive_lambdas']]
            lambda_values = [m['lambda_proto'] for m in training_history['progressive_lambdas']]
            plt.plot(lambda_rounds, lambda_values, 'purple', label='Lambda Proto', linewidth=2, marker='o', markersize=4)
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Lambda Proto', fontsize=12)
            plt.title('Progressive Prototype Alignment', fontsize=14, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
        
        # 绘制自适应权重变化（如果有）
        if training_history['adaptive_weights'] and len(training_history['adaptive_weights']) > 0:
            plt.subplot(2, 3, 5)
            weight_rounds = [m['round'] for m in training_history['adaptive_weights']]
            for client_id in selected_clients:
                weights_per_client = [
                    m['weights'].get(client_id, 0.0) 
                    for m in training_history['adaptive_weights']
                    if client_id in m['weights']
                ]
                if weights_per_client:
                    plt.plot(weight_rounds[:len(weights_per_client)], weights_per_client, 
                           label=f'Client {client_id}', linewidth=2, marker='s', markersize=3)
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Adaptive Weight', fontsize=12)
            plt.title('Adaptive Client Weights', fontsize=14, fontweight='bold')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        # 绘制自适应学习率变化（如果有）
        if training_history['adaptive_lrs']:
            plt.subplot(2, 3, 6)
            lr_data = {}
            for entry in training_history['adaptive_lrs']:
                client_id = entry['client_id']
                if client_id not in lr_data:
                    lr_data[client_id] = {'rounds': [], 'lrs': []}
                lr_data[client_id]['rounds'].append(entry['round'])
                lr_data[client_id]['lrs'].append(entry['learning_rate'])
            
            for client_id, data in lr_data.items():
                plt.plot(data['rounds'], data['lrs'], label=f'Client {client_id}', 
                        linewidth=2, marker='^', markersize=3)
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Learning Rate', fontsize=12)
            plt.title('Adaptive Learning Rates', fontsize=14, fontweight='bold')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = run_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to: {plot_path}")
    
    # 保存配置
    config_path = run_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 打印最终总结
    if training_history['eval_metrics']:
        final_metrics = training_history['eval_metrics'][-1]
        print(f"\nFinal Results:")
        print(f"  Average AUC: {final_metrics['avg_auc']:.4f}")
        print(f"  Average F1: {final_metrics['avg_f1']:.4f}")
        print(f"  Results saved to: {run_dir}")


if __name__ == "__main__":
    main()

