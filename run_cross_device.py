"""
Cross-Device Federated Learning Runner for Fed-ProFiLA-AD
跨设备联邦学习主运行脚本
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
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.seeding import set_seed
from dataset_loader.cross_device_dataset import (
    create_cross_device_dataloader, 
    get_cross_device_client_ids,
    get_device_type_statistics
)
from models.backbone_cnn import create_backbone
from models.device_aware_adapters import get_device_type_id
from methods.fed_profila_ad import (
    compute_total_loss,
    aggregate_models,
    aggregate_prototypes,
    initialize_global_prototype
)
from trainers.server_loop import Server
from trainers.client_loop import Client
from eval.inference import evaluate_all_clients
from eval.metrics import calculate_auc, calculate_f1_score

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/cross_device_fed_profila_ad.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CrossDeviceFederationRunner:
    """
    跨设备联邦学习运行器
    """
    
    def __init__(self, config_path: str):
        """
        初始化跨设备联邦学习运行器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 设置随机种子
        set_seed(self.config['system']['seed'])
        
        # 设置设备
        self.device = torch.device(self.config['system']['device'])
        
        # 初始化客户端信息
        self.client_infos = get_cross_device_client_ids(self.config['dataset']['root_path'])
        self.device_stats = get_device_type_statistics(self.config['dataset']['root_path'])
        
        # 打印统计信息
        self._print_dataset_statistics()
        
        # 初始化模型和训练器
        self._initialize_models()
        self._initialize_trainers()
        
        # 初始化训练历史记录
        self.training_history = {
            'round_losses': [],
            'round_metrics': [],
            'client_metrics': {},
            'global_prototype_history': [],
            'timestamps': []
        }
        
        # 创建输出目录
        self.output_dir = Path(self.config.get('output', {}).get('save_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Cross-device federation runner initialized successfully")
        logger.info(f"Output directory: {self.run_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _print_dataset_statistics(self):
        """打印数据集统计信息"""
        logger.info("=== Dataset Statistics ===")
        total_normal = 0
        total_abnormal = 0
        
        for client_id, stats in self.device_stats.items():
            logger.info(f"Client {client_id} ({stats['device_type']}): "
                       f"{stats['normal_samples']} normal, "
                       f"{stats['abnormal_samples']} abnormal, "
                       f"rate: {stats['anomaly_rate']:.3f}")
            total_normal += stats['normal_samples']
            total_abnormal += stats['abnormal_samples']
        
        total_samples = total_normal + total_abnormal
        overall_rate = total_abnormal / total_samples if total_samples > 0 else 0.0
        
        logger.info(f"Total: {total_normal} normal, {total_abnormal} abnormal, "
                   f"overall rate: {overall_rate:.3f}")
        logger.info("=" * 30)
    
    def _initialize_models(self):
        """初始化模型"""
        model_config = self.config['model']
        
        # 创建骨干网络
        self.backbone = create_backbone(
            backbone_type=model_config['backbone_type'],
            input_channels=model_config['input_channels'],
            feature_dim=model_config['feature_dim'],
            prototype_dim=model_config['prototype_dim'],
            dropout_rate=model_config['dropout_rate']
        ).to(self.device)
        
        # 获取特征维度
        self.feature_dim = self.backbone.get_feature_dim()
        self.prototype_dim = self.backbone.get_prototype_dim()
        
        # 初始化全局原型
        self.global_prototype = initialize_global_prototype(
            self.prototype_dim, self.device
        )
        
        logger.info(f"Model initialized: feature_dim={self.feature_dim}, "
                   f"prototype_dim={self.prototype_dim}")
    
    def _initialize_trainers(self):
        """初始化训练器"""
        # 初始化服务器
        self.server = Server(
            global_model=self.backbone,
            device=self.device,
            client_selection_strategy=self.config['federation']['client_selection'],
            client_selection_fraction=self.config['federation'].get('selection_fraction', 1.0),
            feature_dim=self.feature_dim
        )
        
        # 初始化客户端
        self.clients = {}
        for client_info in self.client_infos:
            client_id = client_info['client_id']
            
            # 创建数据加载器
            train_loader = create_cross_device_dataloader(
                root_path=self.config['dataset']['root_path'],
                client_id=client_id,
                is_train=True,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['system']['num_workers'],
                pin_memory=self.config['system']['pin_memory'],
                **self.config['dataset']['audio']
            )
            
            test_loader = create_cross_device_dataloader(
                root_path=self.config['dataset']['root_path'],
                client_id=client_id,
                is_train=False,
                batch_size=self.config['training']['batch_size'],
                num_workers=self.config['system']['num_workers'],
                pin_memory=self.config['system']['pin_memory'],
                **self.config['dataset']['audio']
            )
            
            # 检查数据集是否为空
            if train_loader is None or test_loader is None:
                logger.warning(f"Client {client_id} has empty dataset, skipping...")
                continue
            
            train_size = len(train_loader.dataset)
            test_size = len(test_loader.dataset)
            
            if train_size == 0:
                logger.warning(f"Client {client_id} has no training data, skipping...")
                continue
            
            if test_size == 0:
                logger.warning(f"Client {client_id} has no test data, skipping...")
                continue
            
            # 创建客户端
            self.clients[client_id] = Client(
                client_id=client_id,
                model=self.backbone,
                train_loader=train_loader,
                test_loader=test_loader,
                device=self.device,
                learning_rate=self.config['training']['learning_rate'],
                optimizer_name=self.config['training']['optimizer'],
                weight_decay=self.config['training']['weight_decay']
            )
            
            logger.info(f"Client {client_id} initialized with "
                       f"{train_size} train samples, "
                       f"{test_size} test samples")
    
    def run_federation(self):
        """运行联邦学习"""
        logger.info("Starting cross-device federated learning...")
        
        # 获取联邦学习配置
        fed_config = self.config['federation']
        num_rounds = fed_config['num_rounds']
        eval_frequency = self.config['evaluation']['eval_frequency']
        
        # 训练循环
        for round_idx in range(num_rounds):
            print("\n" + "="*80)
            print(f"联邦学习第 {round_idx + 1} 轮 / 共 {num_rounds} 轮")
            print("="*80)
            logger.info(f"=== Round {round_idx + 1}/{num_rounds} ===")
            
            # 选择客户端
            selected_clients = self._select_clients(round_idx)
            print(f"参与客户端: {[c.client_id for c in selected_clients]}")
            logger.info(f"Selected clients: {[c.client_id for c in selected_clients]}")
            
            # 客户端训练
            client_results = []
            print(f"开始训练 {len(selected_clients)} 个客户端...")
            for i, client in enumerate(selected_clients, 1):
                print(f"  训练客户端 {i}/{len(selected_clients)}: {client.client_id}")
                logger.info(f"Training client {client.client_id}...")
                
                # 获取当前全局状态
                global_state = self.server.get_global_model_state()
                
                # 客户端训练
                model_state, local_prototype = client.train(
                    global_state,
                    self.server.get_global_prototype()
                )
                
                # 获取客户端对应的设备类型
                client_info = next((c for c in self.client_infos if c['client_id'] == client.client_id), None)
                device_type = client_info['device_type'] if client_info else 'unknown'
                
                client_results.append({
                    'client_id': client.client_id,
                    'model_state': model_state,
                    'local_prototype': local_prototype,
                    'device_type': device_type
                })
                
                logger.info(f"Client {client.client_id} training completed")
            
            # 服务器聚合
            print(f"聚合 {len(client_results)} 个客户端的模型和原型...")
            logger.info("Aggregating models and prototypes...")
            
            # 准备聚合数据
            client_models = {r['client_id']: r['model_state'] for r in client_results}
            client_prototypes = {r['client_id']: r['local_prototype'] for r in client_results}
            client_weights = {r['client_id']: 1.0 / len(client_results) for r in client_results}
            
            # 聚合模型
            self.server.aggregate_models(client_models, client_weights)
            
            # 聚合原型
            self.server.aggregate_prototypes(client_prototypes, client_weights)
            
            # 记录训练历史
            round_avg_loss = np.mean([r.get('train_loss', 0) for r in client_results])
            self.training_history['round_losses'].append(round_avg_loss)
            self.training_history['global_prototype_history'].append(
                self.server.get_global_prototype().cpu().detach().numpy()
            )
            self.training_history['timestamps'].append(datetime.now().isoformat())
            
            # 定期评估
            if (round_idx + 1) % eval_frequency == 0:
                print(f"评估第 {round_idx + 1} 轮性能...")
                logger.info("Running evaluation...")
                eval_metrics = self._evaluate_round(round_idx + 1)
                self.training_history['round_metrics'].append({
                    'round': round_idx + 1,
                    'metrics': eval_metrics
                })
            
            # 定期保存检查点
            if (round_idx + 1) % self.config.get('logging', {}).get('checkpoint_frequency', 10) == 0:
                self._save_checkpoint(round_idx + 1)
            
            # 记录日志
            if (round_idx + 1) % self.config.get('logging', {}).get('log_frequency', 1) == 0:
                self._log_round_info(round_idx + 1)
        
        print("\n" + "="*50)
        print("联邦学习训练完成！")
        print("="*50)
        logger.info("Federated learning completed!")
        
        # 最终评估
        logger.info("Running final evaluation...")
        final_metrics = self._final_evaluation()
        
        # 保存最终模型和结果
        logger.info("Saving final model and results...")
        self._save_final_results(final_metrics)
        
        # 生成可视化
        logger.info("Generating visualizations...")
        self._generate_visualizations()
        
        logger.info(f"All results saved to: {self.run_dir}")
    
    def _select_clients(self, round_idx: int) -> List[Client]:
        """选择参与训练的客户端"""
        fed_config = self.config['federation']
        selection_method = fed_config['client_selection']
        
        if selection_method == "all":
            return list(self.clients.values())
        elif selection_method == "random":
            import random
            selection_fraction = fed_config['selection_fraction']
            num_selected = max(1, int(len(self.clients) * selection_fraction))
            return random.sample(list(self.clients.values()), num_selected)
        elif selection_method == "device_type_balanced":
            return self._select_balanced_clients()
        else:
            raise ValueError(f"Unknown client selection method: {selection_method}")
    
    def _select_balanced_clients(self) -> List[Client]:
        """选择平衡的客户端（确保每种设备类型都有代表）"""
        cross_device_config = self.config['federation']['cross_device']['device_type_balancing']
        
        if not cross_device_config['enabled']:
            return list(self.clients.values())
        
        selected_clients = []
        
        # 选择风扇设备客户端
        fan_clients = [self.clients[cid] for cid in cross_device_config['fan_clients'] 
                      if cid in self.clients]
        if fan_clients:
            selected_clients.extend(fan_clients)
        
        # 选择阀门设备客户端
        valve_clients = [self.clients[cid] for cid in cross_device_config['valve_clients'] 
                        if cid in self.clients]
        if valve_clients:
            selected_clients.extend(valve_clients)
        
        return selected_clients
    
    def _evaluate_round(self, round_idx: int):
        """评估当前轮次"""
        logger.info(f"Evaluating round {round_idx}...")
        
        # 获取当前全局状态
        global_model_state = self.server.get_global_model_state()
        global_prototype = self.server.get_global_prototype()
        
        # 评估所有客户端
        results = {}
        for client_id, client in self.clients.items():
            try:
                # 在客户端上评估
                client_results = client.evaluate(
                    global_model_state,
                    global_prototype
                )
                results[client_id] = client_results
                
                logger.info(f"Client {client_id} - AUC: {client_results.get('auc', 0.0):.4f}, "
                           f"F1: {client_results.get('f1', 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating client {client_id}: {e}")
                results[client_id] = {'auc': 0.0, 'f1': 0.0}
        
        # 计算平均性能
        if results:
            # 正确地从字典中提取auc和f1值
            all_aucs = [r.get('auc', 0.0) for r in results.values()]
            all_f1s = [r.get('f1', 0.0) for r in results.values()]  # 使用 .get() 避免KeyError

            avg_auc = sum(all_aucs) / len(all_aucs) if all_aucs else 0.0
            avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0

            print(f"第 {round_idx} 轮评估 - 平均 AUC: {avg_auc:.4f}, 平均 F1: {avg_f1:.4f}")
            logger.info(f"Round {round_idx} - Average AUC: {avg_auc:.4f}, "
                       f"Average F1: {avg_f1:.4f}")
        else:
            avg_auc = 0.0
            avg_f1 = 0.0
            print(f"第 {round_idx} 轮评估 - 没有有效的评估结果")
            logger.warning(f"Round {round_idx} - No valid evaluation results found")

        # 将整个客户端指标字典存储起来，而不仅仅是auc和f1
        self.training_history['client_metrics'][f'round_{round_idx}'] = results
        
        return {'average_auc': avg_auc, 'average_f1': avg_f1}
    
    def _log_round_info(self, round_idx: int):
        """记录轮次信息"""
        global_state = self.server.get_global_model_state()
        global_proto = self.server.get_global_prototype()
        
        print(f"第 {round_idx + 1} 轮完成 - 全局原型范数: {torch.norm(global_proto):.4f}")
        logger.info(f"Round {round_idx} completed - "
                   f"Global prototype norm: {torch.norm(global_proto):.4f}")
    
    def _save_checkpoint(self, round_idx: int):
        """保存检查点"""
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'round': round_idx,
            'model_state': self.server.get_global_model_state(),
            'global_prototype': self.server.get_global_prototype().cpu(),
            'training_history': self.training_history,
            'config': self.config
        }
        
        checkpoint_path = checkpoint_dir / f"checkpoint_round_{round_idx}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_results(self, final_metrics: Dict):
        """保存最终结果"""
        # 保存最终模型
        model_path = self.run_dir / "final_model.pt"
        torch.save({
            'model_state': self.server.get_global_model_state(),
            'global_prototype': self.server.get_global_prototype().cpu(),
            'config': self.config
        }, model_path)
        logger.info(f"Final model saved: {model_path}")
        
        # 保存客户端适配器
        adapters_dir = self.run_dir / "client_adapters"
        adapters_dir.mkdir(exist_ok=True)
        for client_id, client in self.clients.items():
            adapter_path = adapters_dir / f"adapter_{client_id}.pt"
            torch.save({
                'adapter_state': client.model.adapter.state_dict(),
                'film_generator_state': client.model.film_generator.state_dict(),
            }, adapter_path)
        logger.info(f"Client adapters saved: {adapters_dir}")
        
        # 保存训练历史为JSON
        history_path = self.run_dir / "training_history.json"
        history_to_save = {
            'round_losses': [float(x) for x in self.training_history['round_losses']],
            'round_metrics': self.training_history['round_metrics'],
            'timestamps': self.training_history['timestamps']
        }
        with open(history_path, 'w') as f:
            json.dump(history_to_save, f, indent=2)
        logger.info(f"Training history saved: {history_path}")
        
        # 保存最终评估指标为CSV
        metrics_data = []
        for client_id, metrics in final_metrics.items():
            metrics_row = {'client_id': client_id}
            metrics_row.update(metrics)
            # 获取设备类型
            client_info = next((c for c in self.client_infos if c['client_id'] == client_id), None)
            metrics_row['device_type'] = client_info['device_type'] if client_info else 'unknown'
            metrics_data.append(metrics_row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_path = self.run_dir / "final_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Final metrics saved: {metrics_path}")
        
        # 保存配置文件
        config_path = self.run_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Config saved: {config_path}")
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        viz_dir = self.run_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 设置中文字体（如果可用）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        # 1. 训练损失曲线
        if self.training_history['round_losses']:
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.training_history['round_losses']) + 1), 
                    self.training_history['round_losses'], 
                    marker='o', linewidth=2, markersize=4)
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Average Loss', fontsize=12)
            plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(viz_dir / "training_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Training loss curve saved")
        
        # 2. 评估指标变化（如果有）
        if self.training_history['round_metrics']:
            rounds = [m['round'] for m in self.training_history['round_metrics']]
            
            # 提取所有客户端的指标
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # AUC曲线
            ax = axes[0, 0]
            for client_id in self.clients.keys():
                aucs = [m['metrics'].get(client_id, {}).get('auc', np.nan) 
                       for m in self.training_history['round_metrics']]
                if not all(np.isnan(aucs)):
                    ax.plot(rounds, aucs, marker='o', label=client_id, linewidth=2)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('AUC', fontsize=12)
            ax.set_title('AUC Score by Client', fontsize=14, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # F1曲线
            ax = axes[0, 1]
            for client_id in self.clients.keys():
                f1s = [m['metrics'].get(client_id, {}).get('f1', np.nan) 
                      for m in self.training_history['round_metrics']]
                if not all(np.isnan(f1s)):
                    ax.plot(rounds, f1s, marker='s', label=client_id, linewidth=2)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.set_title('F1 Score by Client', fontsize=14, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Precision曲线
            ax = axes[1, 0]
            for client_id in self.clients.keys():
                precs = [m['metrics'].get(client_id, {}).get('precision', np.nan) 
                        for m in self.training_history['round_metrics']]
                if not all(np.isnan(precs)):
                    ax.plot(rounds, precs, marker='^', label=client_id, linewidth=2)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Precision by Client', fontsize=14, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Recall曲线
            ax = axes[1, 1]
            for client_id in self.clients.keys():
                recalls = [m['metrics'].get(client_id, {}).get('recall', np.nan) 
                          for m in self.training_history['round_metrics']]
                if not all(np.isnan(recalls)):
                    ax.plot(rounds, recalls, marker='v', label=client_id, linewidth=2)
            ax.set_xlabel('Round', fontsize=12)
            ax.set_ylabel('Recall', fontsize=12)
            ax.set_title('Recall by Client', fontsize=14, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "evaluation_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Evaluation metrics plot saved")
        
        # 3. 按设备类型分组的性能对比
        metrics_path = self.run_dir / "final_metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 按设备类型的AUC对比
            ax = axes[0]
            device_types = df['device_type'].unique()
            x_pos = np.arange(len(df))
            colors = {'fan': 'skyblue', 'valve': 'lightcoral'}
            for i, row in df.iterrows():
                ax.bar(i, row['auc'], color=colors.get(row['device_type'], 'gray'), 
                      alpha=0.7, label=row['device_type'] if i == 0 or row['device_type'] != df.iloc[i-1]['device_type'] else "")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df['client_id'], rotation=45, ha='right')
            ax.set_xlabel('Client', fontsize=12)
            ax.set_ylabel('AUC', fontsize=12)
            ax.set_title('AUC by Client and Device Type', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # 按设备类型的F1对比
            ax = axes[1]
            for i, row in df.iterrows():
                ax.bar(i, row['f1'], color=colors.get(row['device_type'], 'gray'), 
                      alpha=0.7, label=row['device_type'] if i == 0 or row['device_type'] != df.iloc[i-1]['device_type'] else "")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(df['client_id'], rotation=45, ha='right')
            ax.set_xlabel('Client', fontsize=12)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.set_title('F1 Score by Client and Device Type', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "device_type_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Device type comparison plot saved")
        
        logger.info(f"All visualizations saved to: {viz_dir}")
    
    def _final_evaluation(self):
        """最终评估"""
        logger.info("Running final evaluation...")
        
        # 获取最终全局状态
        global_model_state = self.server.get_global_model_state()
        global_prototype = self.server.get_global_prototype()
        
        # 评估所有客户端
        final_results = {}
        for client_id, client in self.clients.items():
            try:
                results = client.evaluate(
                    global_model_state,
                    global_prototype
                )
                final_results[client_id] = results
                
                # 获取设备类型
                client_info = next((c for c in self.client_infos if c['client_id'] == client_id), None)
                device_type = client_info['device_type'] if client_info else 'unknown'
                
                logger.info(f"Final - Client {client_id} ({device_type}): "
                           f"AUC: {results.get('auc', 0.0):.4f}, F1: {results.get('f1', 0.0):.4f}")
                
            except Exception as e:
                logger.error(f"Error in final evaluation for client {client_id}: {e}")
                final_results[client_id] = {'auc': 0.0, 'f1': 0.0}
        
        # 按设备类型分组统计
        device_type_results = {}
        for client_id, results in final_results.items():
            client_info = next((c for c in self.client_infos if c['client_id'] == client_id), None)
            device_type = client_info['device_type'] if client_info else 'unknown'
            if device_type not in device_type_results:
                device_type_results[device_type] = []
            device_type_results[device_type].append(results['auc'])
        
        logger.info("=== Final Results by Device Type ===")
        for device_type, aucs in device_type_results.items():
            avg_auc = sum(aucs) / len(aucs)
            logger.info(f"{device_type.upper()}: Average AUC = {avg_auc:.4f} "
                       f"({len(aucs)} clients)")
        
        # 总体平均
        all_aucs = [r['auc'] for r in final_results.values()]
        overall_auc = sum(all_aucs) / len(all_aucs) if all_aucs else 0.0
        logger.info(f"Overall Average AUC: {overall_auc:.4f}")
        logger.info("=" * 40)
        
        return final_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Cross-Device Federated Learning for Fed-ProFiLA-AD')
    parser.add_argument('--config', type=str, default='configs/cross_device_federation.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        return
    
    # 创建结果目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs('results/cross_device', exist_ok=True)
    
    try:
        # 创建并运行联邦学习
        runner = CrossDeviceFederationRunner(args.config)
        runner.run_federation()
        
        print("\n" + "="*60)
        print("跨设备联邦学习成功完成！")
        print("="*60)
        logger.info("Cross-device federated learning completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during federated learning: {e}")
        raise


if __name__ == "__main__":
    main()
