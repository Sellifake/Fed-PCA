"""
Test script for Advanced Fed-ProFiLA-AD
测试训练好的创新版本模型在测试集上的表现
加载检查点 -> 评估 -> 保存CSV -> 画图
"""

# ============================================================================
# 配置区域 - 请在此处修改参数
# ============================================================================
# 选择训练结果的方式（优先级从高到低）：
# 1. SELECT_INDEX: 通过序号选择（例如：1 表示最新的创新版本训练结果）
# 2. SELECT_TIMESTAMP: 通过时间戳选择（例如："20251101_094647"）
# 3. CHECKPOINT_PATH: 直接指定检查点路径（例如："outputs/advanced_run_20251101_094647/final_model.pt"）
# 注意：如果都设置为None，会自动选择最新的创新版本训练结果

SELECT_INDEX = None  # 通过序号选择，例如：1 表示最新的创新版本
SELECT_TIMESTAMP = None  # 通过时间戳选择，例如："20251101_094647"
CHECKPOINT_PATH = None  # 直接指定完整路径，例如："outputs/advanced_run_20251101_094647/final_model.pt"

# 其他可选配置
CONFIG_PATH = None  # 配置文件路径（可选，默认从检查点读取）
DEVICE = None  # 设备（可选，例如："cuda" 或 "cpu"，默认自动检测）
# ============================================================================

import os
import sys
import torch
import yaml
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.seeding import set_seed
from dataset_loader.base_dataset import create_dataloader, get_client_ids
from models.backbone_cnn import create_backbone
from methods.fed_profila_ad import compute_local_prototype
from eval.inference import evaluate_client

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """
    加载检查点
    
    Args:
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        tuple: (模型状态, 全局原型, 配置, 训练历史)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_state = checkpoint['model_state']
    global_prototype = checkpoint['global_prototype']
    config = checkpoint.get('config', {})
    training_history = checkpoint.get('training_history', {})
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Model state keys: {len(model_state.keys())} parameters")
    logger.info(f"Global prototype shape: {global_prototype.shape}")
    
    # 检查是否有创新功能使用记录
    if training_history.get('adaptive_weights'):
        logger.info("✓ Adaptive client weighting was used in training")
    if training_history.get('progressive_lambdas'):
        logger.info("✓ Progressive prototype alignment was used in training")
    if training_history.get('adaptive_lrs'):
        logger.info("✓ Adaptive learning rate scheduling was used in training")
    
    return model_state, global_prototype, config, training_history


def test_trained_model(checkpoint_path: str, config_path: str = None, device_str: str = None):
    """
    测试训练好的模型
    
    Args:
        checkpoint_path: 检查点路径
        config_path: 配置文件路径（可选）
        device_str: 设备字符串（可选）
    """
    logger.info("=" * 60)
    logger.info("Testing Trained Advanced Fed-ProFiLA-AD Model")
    logger.info("=" * 60)
    
    # 设置设备
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # 加载检查点
    model_state, global_prototype, checkpoint_config, training_history = load_checkpoint(checkpoint_path, device)
    
    # 加载配置
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = checkpoint_config
        logger.info("Using config from checkpoint")
    
    if not config:
        raise ValueError("Configuration not found in checkpoint and no config file provided")
    
    # 获取客户端ID列表
    root_path = config['dataset']['root_path']
    client_ids = get_client_ids(root_path)
    
    if not client_ids:
        logger.error(f"No clients found in {root_path}")
        return
    
    logger.info(f"Found {len(client_ids)} clients: {client_ids}")
    
    # 创建模型
    model_config = config['model']
    backbone = create_backbone(
        backbone_type=model_config['backbone_type'],
        input_channels=model_config['input_channels'],
        feature_dim=model_config['feature_dim'],
        prototype_dim=model_config['prototype_dim'],
        dropout_rate=model_config['dropout_rate'],
        hidden_dims=model_config.get('hidden_dims', [64, 128, 256]),
        use_projection_head=model_config.get('use_projection_head', True)
    ).to(device)
    
    # 加载模型权重
    # 检查点中保存的是共享参数（film_generator 和 encoder），adapter 和 projection 不在检查点中
    # 这些参数会在计算本地原型时自动初始化，属于正常情况
    SHARED_PREFIXES = ("film_generator", "encoder")
    
    # 检查检查点中的参数
    checkpoint_keys = set(model_state.keys())
    logger.info(f"检查点包含 {len(checkpoint_keys)} 个参数")
    
    # 过滤出共享参数（检查点应该只包含共享参数）
    filtered_state = {
        name: param
        for name, param in model_state.items()
        if any(name.startswith(prefix) for prefix in SHARED_PREFIXES)
    }
    
    logger.info(f"过滤后的共享参数: {len(filtered_state)} 个")
    if len(filtered_state) == 0:
        logger.warning("检查点中没有找到共享参数，尝试加载所有参数")
        filtered_state = model_state
    
    # 加载共享参数（strict=False允许缺少adapter和projection）
    missing_keys, unexpected_keys = backbone.load_state_dict(filtered_state, strict=False)
    
    if missing_keys:
        # 过滤出真正缺失的共享参数（不应该缺失的）
        shared_missing = [k for k in missing_keys if any(k.startswith(p) for p in SHARED_PREFIXES)]
        local_missing = [k for k in missing_keys if k not in shared_missing]
        
        if shared_missing:
            logger.warning(f"缺少共享参数（可能有问题）: {len(shared_missing)} 个")
            for key in shared_missing[:5]:
                logger.warning(f"  - {key}")
        
        if local_missing:
            logger.info(f"本地参数未加载（正常情况，adapter和projection是本地参数）: {len(local_missing)} 个")
            logger.info(f"  包括: adapter (本地适配器), projection (投影头)")
    
    if unexpected_keys:
        logger.warning(f"检查点中有未使用的参数: {len(unexpected_keys)} 个")
        for key in list(unexpected_keys)[:5]:
            logger.warning(f"  - {key}")
    
    backbone.eval()
    
    # 全局原型移到设备
    global_prototype = global_prototype.to(device)
    
    # 评估所有客户端
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating all clients on test set...")
    logger.info("=" * 60)
    
    all_results = []
    client_results_dict = {}
    
    dataset_config = config['dataset']
    training_config = config['training']
    
    for client_id in client_ids:
        logger.info(f"\nEvaluating client {client_id}...")
        
        # 创建测试数据加载器
        test_loader = create_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=False,
            batch_size=training_config['batch_size'],
            num_workers=config['system'].get('num_workers', 0),
            pin_memory=False,
            sample_rate=dataset_config['sample_rate'],
            segment_length=dataset_config['segment_length'],
            n_mels=dataset_config['n_mels'],
            hop_length=dataset_config['hop_length'],
            n_fft=dataset_config['n_fft'],
            max_samples=dataset_config.get('max_test_samples', None)
        )
        
        if test_loader is None or len(test_loader.dataset) == 0:
            logger.warning(f"Client {client_id} has no test data, skipping...")
            continue
        
        # 计算本地原型（使用训练数据）
        train_loader = create_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=True,
            batch_size=training_config['batch_size'],
            num_workers=0,
            pin_memory=False,
            sample_rate=dataset_config['sample_rate'],
            segment_length=dataset_config['segment_length'],
            n_mels=dataset_config['n_mels'],
            hop_length=dataset_config['hop_length'],
            n_fft=dataset_config['n_fft'],
            max_samples=dataset_config.get('max_train_samples', None),
            include_abnormal_in_train=dataset_config.get('include_abnormal_in_train', True),
            abnormal_fraction=dataset_config.get('abnormal_fraction', 0.2)
        )
        
        if train_loader is None or len(train_loader.dataset) == 0:
            logger.warning(f"Client {client_id} has no training data for prototype computation, skipping...")
            continue
        
        # 计算本地原型
        local_prototype = compute_local_prototype(
            model=backbone,
            dataloader=train_loader,
            global_prototype=global_prototype,
            device=device
        )
        
        # 评估客户端
        metrics = evaluate_client(
            model=backbone,
            test_loader=test_loader,
            global_prototype=global_prototype,
            local_prototype=local_prototype,
            device=device,
            client_id=client_id
        )
        
        client_results_dict[client_id] = metrics
        
        # 保存结果
        result_row = {
            'client_id': client_id,
            'auc': metrics.get('auc', 0.0),
            'pr_auc': metrics.get('pr_auc', 0.0),
            'f1': metrics.get('best_f1', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'specificity': metrics.get('specificity', 0.0),
            'sensitivity': metrics.get('sensitivity', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'best_threshold': metrics.get('best_threshold', 0.0),
            'test_samples': len(test_loader.dataset)
        }
        all_results.append(result_row)
        
        logger.info(f"Client {client_id} - AUC: {metrics.get('auc', 0.0):.4f}, "
                   f"F1: {metrics.get('best_f1', 0.0):.4f}, "
                   f"Precision: {metrics.get('precision', 0.0):.4f}, "
                   f"Recall: {metrics.get('recall', 0.0):.4f}")
    
    if not all_results:
        logger.error("No valid evaluation results!")
        return
    
    # 计算平均指标
    df = pd.DataFrame(all_results)
    avg_metrics = {
        'auc': df['auc'].mean(),
        'pr_auc': df['pr_auc'].mean(),
        'f1': df['f1'].mean(),
        'precision': df['precision'].mean(),
        'recall': df['recall'].mean(),
        'specificity': df['specificity'].mean(),
        'sensitivity': df['sensitivity'].mean(),
        'accuracy': df['accuracy'].mean()
    }
    
    std_metrics = {
        'auc_std': df['auc'].std(),
        'pr_auc_std': df['pr_auc'].std(),
        'f1_std': df['f1'].std(),
        'precision_std': df['precision'].std(),
        'recall_std': df['recall'].std(),
        'specificity_std': df['specificity'].std(),
        'sensitivity_std': df['sensitivity'].std(),
        'accuracy_std': df['accuracy'].std()
    }
    
    # 打印平均结果
    print("\n" + "=" * 60)
    print("AVERAGE METRICS ACROSS ALL CLIENTS")
    print("=" * 60)
    for key, value in avg_metrics.items():
        std_key = f"{key}_std"
        std_value = std_metrics.get(std_key, 0.0)
        print(f"{key.upper()}: {value:.4f} ± {std_value:.4f}")
    
    # 添加平均行
    avg_row = {'client_id': 'AVERAGE'}
    avg_row.update(avg_metrics)
    avg_row.update(std_metrics)
    avg_row['best_threshold'] = df['best_threshold'].mean()
    avg_row['test_samples'] = df['test_samples'].sum()
    all_results.append(avg_row)
    
    # 保存结果到CSV
    checkpoint_dir = Path(checkpoint_path).parent
    csv_path = checkpoint_dir / "test_results.csv"
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(csv_path, index=False)
    logger.info(f"\nTest results saved to: {csv_path}")
    
    # 绘制评估结果图表（包含创新功能分析）
    plot_path = checkpoint_dir / "test_results_visualization.png"
    plot_test_results(df, avg_metrics, plot_path, training_history)
    logger.info(f"Visualization saved to: {plot_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing completed successfully!")
    logger.info("=" * 60)


def plot_test_results(df: pd.DataFrame, avg_metrics: Dict, save_path: Path, training_history: Dict = None):
    """
    绘制测试结果可视化图表（包含创新功能分析）
    
    Args:
        df: 结果DataFrame
        avg_metrics: 平均指标字典
        save_path: 保存路径
        training_history: 训练历史（用于显示创新功能效果）
    """
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    if training_history and (training_history.get('adaptive_weights') or 
                             training_history.get('progressive_lambdas') or 
                             training_history.get('adaptive_lrs')):
        # 如果有创新功能数据，使用更大的图
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 各客户端AUC和F1对比
        ax1 = fig.add_subplot(gs[0, 0])
        x_pos = np.arange(len(df))
        width = 0.35
        ax1.bar(x_pos - width/2, df['auc'], width, label='AUC', alpha=0.8, color='skyblue')
        ax1.bar(x_pos + width/2, df['f1'], width, label='F1 Score', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Client', fontsize=10)
        ax1.set_ylabel('Score', fontsize=10)
        ax1.set_title('AUC and F1 Score by Client', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df['client_id'], rotation=45, ha='right', fontsize=8)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.1])
        
        # 2. Precision和Recall对比
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.bar(x_pos - width/2, df['precision'], width, label='Precision', alpha=0.8, color='lightgreen')
        ax2.bar(x_pos + width/2, df['recall'], width, label='Recall', alpha=0.8, color='orange')
        ax2.set_xlabel('Client', fontsize=10)
        ax2.set_ylabel('Score', fontsize=10)
        ax2.set_title('Precision and Recall by Client', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df['client_id'], rotation=45, ha='right', fontsize=8)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.1])
        
        # 3. 平均指标汇总
        ax3 = fig.add_subplot(gs[0, 2])
        metrics_names = ['AUC', 'F1', 'Precision', 'Recall']
        metrics_values = [
            avg_metrics['auc'],
            avg_metrics['f1'],
            avg_metrics['precision'],
            avg_metrics['recall']
        ]
        y_pos = np.arange(len(metrics_names))
        ax3.barh(y_pos, metrics_values, alpha=0.8, color='steelblue')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(metrics_names, fontsize=9)
        ax3.set_xlabel('Score', fontsize=10)
        ax3.set_title('Average Metrics', fontsize=12, fontweight='bold')
        ax3.set_xlim([0, 1.1])
        ax3.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(metrics_values):
            ax3.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=9)
        
        # 4. 测试样本数量分布
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.bar(df['client_id'], df['test_samples'], alpha=0.8, color='mediumpurple')
        ax4.set_xlabel('Client', fontsize=10)
        ax4.set_ylabel('Test Samples', fontsize=10)
        ax4.set_title('Test Samples Distribution', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45, labelsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 渐进式lambda_proto变化（如果有）
        if training_history.get('progressive_lambdas'):
            ax5 = fig.add_subplot(gs[1, 1])
            lambda_data = training_history['progressive_lambdas']
            rounds = [m['round'] for m in lambda_data]
            lambdas = [m['lambda_proto'] for m in lambda_data]
            ax5.plot(rounds, lambdas, 'purple', linewidth=2, marker='o', markersize=4)
            ax5.set_xlabel('Round', fontsize=10)
            ax5.set_ylabel('Lambda Proto', fontsize=10)
            ax5.set_title('Progressive Prototype Alignment', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. 自适应权重变化（如果有）
        if training_history.get('adaptive_weights'):
            ax6 = fig.add_subplot(gs[1, 2])
            weight_data = training_history['adaptive_weights']
            rounds = [m['round'] for m in weight_data]
            for client_id in df['client_id']:
                weights_per_client = [
                    m['weights'].get(client_id, 0.0) 
                    for m in weight_data
                    if client_id in m['weights']
                ]
                if weights_per_client:
                    ax6.plot(rounds[:len(weights_per_client)], weights_per_client, 
                           label=f'Client {client_id}', linewidth=1.5, marker='s', markersize=3)
            ax6.set_xlabel('Round', fontsize=10)
            ax6.set_ylabel('Adaptive Weight', fontsize=10)
            ax6.set_title('Adaptive Client Weights', fontsize=12, fontweight='bold')
            ax6.legend(fontsize=7)
            ax6.grid(True, alpha=0.3)
        
        # 7. 自适应学习率变化（如果有）
        if training_history.get('adaptive_lrs'):
            ax7 = fig.add_subplot(gs[2, 0:])
            lr_data = training_history['adaptive_lrs']
            lr_dict = {}
            for entry in lr_data:
                client_id = entry['client_id']
                if client_id not in lr_dict:
                    lr_dict[client_id] = {'rounds': [], 'lrs': []}
                lr_dict[client_id]['rounds'].append(entry['round'])
                lr_dict[client_id]['lrs'].append(entry['learning_rate'])
            
            for client_id, data in lr_dict.items():
                ax7.plot(data['rounds'], data['lrs'], label=f'Client {client_id}', 
                        linewidth=1.5, marker='^', markersize=3)
            ax7.set_xlabel('Round', fontsize=10)
            ax7.set_ylabel('Learning Rate', fontsize=10)
            ax7.set_title('Adaptive Learning Rate Scheduling', fontsize=12, fontweight='bold')
            ax7.legend(fontsize=8, ncol=min(3, len(lr_dict)))
            ax7.grid(True, alpha=0.3)
        
    else:
        # 标准图表（如果没有创新功能数据）
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 各客户端AUC和F1对比
        ax = axes[0, 0]
        x_pos = np.arange(len(df))
        width = 0.35
        ax.bar(x_pos - width/2, df['auc'], width, label='AUC', alpha=0.8, color='skyblue')
        ax.bar(x_pos + width/2, df['f1'], width, label='F1 Score', alpha=0.8, color='lightcoral')
        ax.set_xlabel('Client', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('AUC and F1 Score by Client', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['client_id'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # 2. Precision和Recall对比
        ax = axes[0, 1]
        ax.bar(x_pos - width/2, df['precision'], width, label='Precision', alpha=0.8, color='lightgreen')
        ax.bar(x_pos + width/2, df['recall'], width, label='Recall', alpha=0.8, color='orange')
        ax.set_xlabel('Client', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Precision and Recall by Client', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['client_id'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])
        
        # 3. 指标汇总
        ax = axes[1, 0]
        metrics_names = ['AUC', 'F1', 'Precision', 'Recall', 'Specificity', 'Sensitivity', 'Accuracy']
        metrics_values = [
            avg_metrics['auc'],
            avg_metrics['f1'],
            avg_metrics['precision'],
            avg_metrics['recall'],
            avg_metrics['specificity'],
            avg_metrics['sensitivity'],
            avg_metrics['accuracy']
        ]
        y_pos = np.arange(len(metrics_names))
        ax.barh(y_pos, metrics_values, alpha=0.8, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics_names)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Average Metrics Summary', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='x')
        for i, v in enumerate(metrics_values):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=10)
        
        # 4. 测试样本数量分布
        ax = axes[1, 1]
        ax.bar(df['client_id'], df['test_samples'], alpha=0.8, color='mediumpurple')
        ax.set_xlabel('Client', fontsize=12)
        ax.set_ylabel('Number of Test Samples', fontsize=12)
        ax.set_title('Test Samples Distribution', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def find_available_checkpoints(outputs_dir: str = "outputs", run_type_filter: str = "advanced"):
    """
    自动查找可用的检查点文件
    
    Args:
        outputs_dir: 输出目录路径
        run_type_filter: 过滤类型，"basic" 只返回基础版本，"advanced" 只返回创新版本
        
    Returns:
        list: 检查点信息列表
    """
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return []
    
    checkpoints = []
    
    # 扫描所有文件夹
    for run_dir in outputs_path.iterdir():
        if not run_dir.is_dir():
            continue
        
        run_name = run_dir.name
        
        # 判断是基础版本还是创新版本
        if run_name.startswith('advanced_run_'):
            run_type = 'advanced'
            timestamp = run_name.replace('advanced_run_', '')
            # 只保留符合过滤类型的
            if run_type_filter == "advanced":
                pass  # 保留
            else:
                continue  # 跳过创新版本
        elif run_name.startswith('run_'):
            run_type = 'basic'
            timestamp = run_name.replace('run_', '')
            # 只保留符合过滤类型的
            if run_type_filter == "basic":
                pass  # 保留
            else:
                continue  # 跳过基础版本
        else:
            continue
        
        # 查找检查点文件
        final_model = run_dir / "final_model.pt"
        best_model = run_dir / "best_model.pt"
        
        if final_model.exists():
            checkpoints.append({
                'run_name': run_name,
                'checkpoint_path': str(final_model),
                'run_type': run_type,
                'timestamp': timestamp,
                'model_type': 'final'
            })
        elif best_model.exists():
            checkpoints.append({
                'run_name': run_name,
                'checkpoint_path': str(best_model),
                'run_type': run_type,
                'timestamp': timestamp,
                'model_type': 'best'
            })
    
    # 按时间戳排序（最新的在前）
    checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return checkpoints


def list_available_checkpoints():
    """列出所有可用的检查点（只显示创新版本）"""
    checkpoints = find_available_checkpoints(run_type_filter="advanced")
    
    if not checkpoints:
        print("\n未找到可用的创新版本检查点文件！")
        print("请确保 outputs/ 目录下有创新版本的训练结果（advanced_run_xxx格式）。")
        return
    
    print("\n" + "=" * 80)
    print("可用的创新版本训练结果：")
    print("=" * 80)
    print(f"{'序号':<6} {'训练时间':<20} {'模型类型':<10} {'路径'}")
    print("-" * 80)
    
    for idx, cp in enumerate(checkpoints, 1):
        model_type_name = "最终模型" if cp['model_type'] == 'final' else "最佳模型"
        print(f"{idx:<6} {cp['timestamp']:<20} {model_type_name:<10} {cp['checkpoint_path']}")
    
    print("=" * 80)
    print(f"\n使用方法:")
    print(f"  修改文件顶部的 SELECT_INDEX, SELECT_TIMESTAMP 或 CHECKPOINT_PATH 配置")
    print(f"  或者: python test_advanced.py --select <序号>")
    print(f"  或者: python test_advanced.py --checkpoint <完整路径>")
    print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Test Advanced Fed-ProFiLA-AD Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接运行（使用文件顶部配置）
  python test_advanced.py
  
  # 列出所有可用的创新版本训练结果
  python test_advanced.py --list
  
  # 选择序号为1的训练结果进行测试（命令行覆盖文件配置）
  python test_advanced.py --select 1
  
  # 直接指定检查点路径（命令行覆盖文件配置）
  python test_advanced.py --checkpoint outputs/advanced_run_20251101_094647/final_model.pt
        """
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (overrides CHECKPOINT_PATH in config)')
    parser.add_argument('--select', type=int, default=None,
                       help='Select checkpoint by index (overrides SELECT_INDEX in config)')
    parser.add_argument('--list', action='store_true',
                       help='List all available advanced version checkpoints')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (overrides CONFIG_PATH in config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (overrides DEVICE in config)')
    
    args = parser.parse_args()
    
    # 列出所有可用检查点
    if args.list:
        list_available_checkpoints()
        return
    
    # 确定要使用的参数（命令行参数优先，然后是文件配置，最后是默认值）
    checkpoint_path = None
    config_path = args.config if args.config is not None else CONFIG_PATH
    device_str = args.device if args.device is not None else DEVICE
    
    # 确定检查点路径
    if args.checkpoint:
        # 命令行指定
        checkpoint_path = args.checkpoint
    elif args.select is not None:
        # 命令行选择序号
        checkpoints = find_available_checkpoints(run_type_filter="advanced")
        if not checkpoints:
            logger.error("未找到可用的创新版本检查点文件！")
            list_available_checkpoints()
            return
        
        if args.select < 1 or args.select > len(checkpoints):
            logger.error(f"无效的序号！请选择 1-{len(checkpoints)} 之间的数字。")
            list_available_checkpoints()
            return
        
        selected = checkpoints[args.select - 1]
        checkpoint_path = selected['checkpoint_path']
        logger.info(f"已选择（命令行）: {selected['run_name']}")
        logger.info(f"检查点路径: {checkpoint_path}")
    elif SELECT_INDEX is not None:
        # 文件配置选择序号
        checkpoints = find_available_checkpoints(run_type_filter="advanced")
        if not checkpoints:
            logger.error("未找到可用的创新版本检查点文件！")
            list_available_checkpoints()
            return
        
        if SELECT_INDEX < 1 or SELECT_INDEX > len(checkpoints):
            logger.error(f"配置中的序号无效！请选择 1-{len(checkpoints)} 之间的数字。")
            list_available_checkpoints()
            return
        
        selected = checkpoints[SELECT_INDEX - 1]
        checkpoint_path = selected['checkpoint_path']
        logger.info(f"已选择（配置序号{SELECT_INDEX}）: {selected['run_name']}")
        logger.info(f"检查点路径: {checkpoint_path}")
    elif SELECT_TIMESTAMP:
        # 文件配置选择时间戳
        checkpoints = find_available_checkpoints(run_type_filter="advanced")
        matching = [cp for cp in checkpoints if cp['timestamp'] == SELECT_TIMESTAMP]
        if not matching:
            logger.error(f"未找到时间戳为 {SELECT_TIMESTAMP} 的创新版本训练结果！")
            list_available_checkpoints()
            return
        
        selected = matching[0]
        checkpoint_path = selected['checkpoint_path']
        logger.info(f"已选择（配置时间戳）: {selected['run_name']}")
        logger.info(f"检查点路径: {checkpoint_path}")
    elif CHECKPOINT_PATH:
        # 文件配置直接指定路径
        checkpoint_path = CHECKPOINT_PATH
        logger.info(f"使用配置路径: {checkpoint_path}")
    else:
        # 默认：选择最新的创新版本
        checkpoints = find_available_checkpoints(run_type_filter="advanced")
        if not checkpoints:
            logger.error("未找到可用的创新版本检查点文件！")
            logger.error("请确保 outputs/ 目录下有创新版本的训练结果（advanced_run_xxx格式），")
            logger.error("或在文件顶部配置 SELECT_INDEX, SELECT_TIMESTAMP 或 CHECKPOINT_PATH")
            list_available_checkpoints()
            return
        
        selected = checkpoints[0]  # 最新的
        checkpoint_path = selected['checkpoint_path']
        logger.info(f"自动选择最新的创新版本: {selected['run_name']}")
        logger.info(f"检查点路径: {checkpoint_path}")
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        logger.error(f"检查点文件不存在: {checkpoint_path}")
        list_available_checkpoints()
        return
    
    try:
        test_trained_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device_str=device_str
        )
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
