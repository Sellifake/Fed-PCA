"""
Cross-Device Dataset Loader for Fed-ProFiLA-AD
实现跨设备联邦学习的数据集加载器，支持MIMII-DUE和MIMII-DEV数据集
"""

import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import glob
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CrossDeviceMIMIIDataset(Dataset):
    """
    跨设备MIMII数据集类
    支持MIMII-DUE和MIMII-DEV数据集的统一加载
    """
    
    def __init__(
        self, 
        root_path: str, 
        client_id: str, 
        is_train: bool = True,
        sample_rate: int = 16000,
        segment_length: int = 4096,
        n_mels: int = 128,
        hop_length: int = 512,
        n_fft: int = 1024,
        device_type: str = None
    ):
        """
        初始化跨设备MIMII数据集
        
        Args:
            root_path: 数据集根目录路径
            client_id: 客户端ID (如 "id_00", "dev_fan", "dev_valve" 等)
            is_train: 是否为训练模式
            sample_rate: 音频采样率
            segment_length: 音频片段长度
            n_mels: Mel频谱图频率bins数量
            hop_length: 频谱图hop长度
            n_fft: FFT窗口大小
            device_type: 设备类型 ("fan" 或 "valve")
        """
        self.root_path = root_path
        self.client_id = client_id
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.device_type = device_type or self._infer_device_type(client_id)
        
        # 构建文件路径
        self.client_path = self._get_client_path()
        
        # 初始化文件列表和标签
        self.file_paths = []
        self.labels = []
        
        # 加载数据
        self._load_data()
        
        # 初始化音频变换
        self._init_transforms()
        
        logger.info(f"Loaded {len(self.file_paths)} samples for client {client_id} "
                   f"({'train' if is_train else 'test'} mode, device_type: {self.device_type})")
    
    def _infer_device_type(self, client_id: str) -> str:
        """
        根据客户端ID推断设备类型
        
        Args:
            client_id: 客户端ID
            
        Returns:
            str: 设备类型
        """
        if client_id.startswith('id_'):
            return "fan"  # DUE数据集都是风扇
        elif client_id == "dev_fan":
            return "fan"
        elif client_id == "dev_valve":
            return "valve"
        else:
            logger.warning(f"Unknown client_id {client_id}, defaulting to fan")
            return "fan"
    
    def _get_client_path(self) -> str:
        """
        获取客户端数据路径
        
        Returns:
            str: 客户端数据路径
        """
        if self.client_id.startswith('id_'):
            # DUE数据集
            return os.path.join(self.root_path, "mimii_due_fan_0db", self.client_id)
        elif self.client_id == "dev_fan":
            # DEV Fan数据集
            return os.path.join(self.root_path, "mimii_dev_fan")
        elif self.client_id == "dev_valve":
            # DEV Valve数据集
            return os.path.join(self.root_path, "mimii_dev_valve")
        else:
            raise ValueError(f"Unknown client_id: {self.client_id}")
    
    def _load_data(self):
        """加载数据文件路径和标签"""
        if self.client_id.startswith('id_'):
            # DUE数据集：按normal/abnormal组织
            self._load_due_data()
        else:
            # DEV数据集：按train/test组织
            self._load_dev_data()
    
    def _load_due_data(self):
        """加载DUE数据集"""
        # 加载Normal数据
        normal_path = os.path.join(self.client_path, "normal")
        if os.path.exists(normal_path):
            normal_files = [f for f in os.listdir(normal_path) if f.endswith('.wav')]
            normal_files.sort()
            
            for file in normal_files:
                self.file_paths.append(os.path.join(normal_path, file))
                self.labels.append(0)  # 0表示Normal
        
        # 如果不是训练模式，加载Abnormal数据
        if not self.is_train:
            abnormal_path = os.path.join(self.client_path, "abnormal")
            if os.path.exists(abnormal_path):
                abnormal_files = [f for f in os.listdir(abnormal_path) if f.endswith('.wav')]
                abnormal_files.sort()
                
                for file in abnormal_files:
                    self.file_paths.append(os.path.join(abnormal_path, file))
                    self.labels.append(1)  # 1表示Abnormal
    
    def _load_dev_data(self):
        """加载DEV数据集
        MIMII-DEV数据集结构：
        - train/ 目录：所有文件都是normal（训练用）
        - test/ 目录：所有文件都是abnormal（测试用）
        """
        if self.is_train:
            # 训练模式：只加载train目录下的所有文件（都是normal）
            train_path = os.path.join(self.client_path, "train")
            if os.path.exists(train_path):
                # 获取所有.wav文件（可能在子目录中）
                for root, dirs, files in os.walk(train_path):
                    for file in files:
                        if file.endswith('.wav'):
                            self.file_paths.append(os.path.join(root, file))
                            self.labels.append(0)  # 0表示Normal
                
                # 排序以保持一致性
                sorted_pairs = sorted(zip(self.file_paths, self.labels))
                self.file_paths, self.labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
                self.file_paths = list(self.file_paths)
                self.labels = list(self.labels)
        else:
            # 测试模式：加载train目录下的normal和test目录下的abnormal
            # 加载Normal数据（train目录）
            train_path = os.path.join(self.client_path, "train")
            if os.path.exists(train_path):
                for root, dirs, files in os.walk(train_path):
                    for file in files:
                        if file.endswith('.wav'):
                            self.file_paths.append(os.path.join(root, file))
                            self.labels.append(0)  # 0表示Normal
            
            # 加载Abnormal数据（test目录）
            test_path = os.path.join(self.client_path, "test")
            if os.path.exists(test_path):
                for root, dirs, files in os.walk(test_path):
                    for file in files:
                        if file.endswith('.wav'):
                            self.file_paths.append(os.path.join(root, file))
                            self.labels.append(1)  # 1表示Abnormal
            
            # 排序以保持一致性
            sorted_pairs = sorted(zip(self.file_paths, self.labels))
            self.file_paths, self.labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
            self.file_paths = list(self.file_paths)
            self.labels = list(self.labels)
    
    def _init_transforms(self):
        """初始化音频变换"""
        # Mel频谱图变换
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # 对数变换
        self.log_transform = T.AmplitudeToDB()
    
    def _load_and_preprocess_audio(self, file_path: str) -> torch.Tensor:
        """
        加载并预处理音频文件
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            torch.Tensor: 预处理后的log-Mel频谱图 [n_mels, time_frames]
        """
        try:
            # 使用librosa加载音频文件
            waveform, sample_rate = librosa.load(file_path, sr=self.sample_rate, mono=True)
            
            # 转换为torch张量
            waveform = torch.from_numpy(waveform).float()
            waveform = waveform.unsqueeze(0)  # 添加batch维度 [1, length]
            
            # 如果音频太短，进行填充
            if waveform.shape[1] < self.segment_length:
                pad_length = self.segment_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_length))
            
            # 如果音频太长，随机裁剪
            elif waveform.shape[1] > self.segment_length:
                if self.is_train:
                    # 训练时随机裁剪
                    start_idx = torch.randint(0, waveform.shape[1] - self.segment_length + 1, (1,)).item()
                else:
                    # 测试时从中间裁剪
                    start_idx = (waveform.shape[1] - self.segment_length) // 2
                
                waveform = waveform[:, start_idx:start_idx + self.segment_length]
            
            # 转换为Mel频谱图
            mel_spec = self.mel_transform(waveform)
            
            # 对数变换
            log_mel_spec = self.log_transform(mel_spec)
            
            # 添加通道维度 [1, n_mels, time_frames]
            log_mel_spec = log_mel_spec.unsqueeze(0)
            
            return log_mel_spec.squeeze(0)  # 移除batch维度 [n_mels, time_frames]
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            # 返回零张量作为fallback
            return torch.zeros(self.n_mels, self.segment_length // self.hop_length)
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.file_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        """
        获取数据项
        
        Args:
            index: 数据索引
            
        Returns:
            Tuple[torch.Tensor, int, str]: (log-Mel频谱图, 标签, 设备类型)
        """
        file_path = self.file_paths[index]
        label = self.labels[index]
        
        # 加载并预处理音频
        spectrogram = self._load_and_preprocess_audio(file_path)
        
        return spectrogram, label, self.device_type


def create_cross_device_dataloader(
    root_path: str,
    client_id: str,
    is_train: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    **dataset_kwargs
) -> Optional[DataLoader]:
    """
    创建跨设备数据加载器
    
    Args:
        root_path: 数据集根目录路径
        client_id: 客户端ID
        is_train: 是否为训练模式
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        pin_memory: 是否使用pin_memory
        **dataset_kwargs: 传递给CrossDeviceMIMIIDataset的其他参数
        
    Returns:
        DataLoader: 数据加载器，如果数据集为空则返回None
    """
    # 创建数据集
    dataset = CrossDeviceMIMIIDataset(
        root_path=root_path,
        client_id=client_id,
        is_train=is_train,
        **dataset_kwargs
    )
    
    # 检查数据集是否为空
    if len(dataset) == 0:
        logger.warning(f"Dataset for client {client_id} ({'train' if is_train else 'test'}) is empty")
        return None
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,  # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_train,  # 训练时丢弃最后一个不完整的batch
        generator=torch.Generator().manual_seed(42) if is_train else None  # 确保可重现性
    )
    
    return dataloader


def get_cross_device_client_ids(root_path: str) -> List[Dict[str, str]]:
    """
    获取所有可用的跨设备客户端信息
    
    Args:
        root_path: 数据集根目录路径
        
    Returns:
        List[Dict[str, str]]: 客户端信息列表，每个字典包含client_id和device_type
    """
    clients = []
    
    # 检查DUE数据集
    due_path = os.path.join(root_path, "mimii_due_fan_0db")
    if os.path.exists(due_path):
        for item in os.listdir(due_path):
            item_path = os.path.join(due_path, item)
            if os.path.isdir(item_path) and item.startswith('id_'):
                clients.append({
                    "client_id": item,
                    "device_type": "fan",
                    "data_source": "mimii_due"
                })
    
    # 检查DEV Fan数据集
    dev_fan_path = os.path.join(root_path, "mimii_dev_fan")
    if os.path.exists(dev_fan_path):
        clients.append({
            "client_id": "dev_fan",
            "device_type": "fan",
            "data_source": "mimii_dev"
        })
    
    # 检查DEV Valve数据集
    dev_valve_path = os.path.join(root_path, "mimii_dev_valve")
    if os.path.exists(dev_valve_path):
        clients.append({
            "client_id": "dev_valve",
            "device_type": "valve",
            "data_source": "mimii_dev"
        })
    
    # 按client_id排序
    clients.sort(key=lambda x: x["client_id"])
    
    return clients


def get_device_type_statistics(root_path: str) -> Dict[str, Dict[str, int]]:
    """
    获取设备类型统计信息
    
    Args:
        root_path: 数据集根目录路径
        
    Returns:
        Dict[str, Dict[str, int]]: 设备类型统计信息
    """
    stats = {}
    clients = get_cross_device_client_ids(root_path)
    
    for client_info in clients:
        client_id = client_info["client_id"]
        device_type = client_info["device_type"]
        
        # 创建数据集获取统计信息
        try:
            train_dataset = CrossDeviceMIMIIDataset(
                root_path=root_path,
                client_id=client_id,
                is_train=True
            )
            test_dataset = CrossDeviceMIMIIDataset(
                root_path=root_path,
                client_id=client_id,
                is_train=False
            )
            
            # 统计Normal和Abnormal样本数
            normal_count = sum(1 for label in train_dataset.labels if label == 0)
            abnormal_count = sum(1 for label in test_dataset.labels if label == 1)
            total_count = normal_count + abnormal_count
            
            stats[client_id] = {
                "device_type": device_type,
                "normal_samples": normal_count,
                "abnormal_samples": abnormal_count,
                "total_samples": total_count,
                "anomaly_rate": abnormal_count / total_count if total_count > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics for client {client_id}: {e}")
            stats[client_id] = {
                "device_type": device_type,
                "normal_samples": 0,
                "abnormal_samples": 0,
                "total_samples": 0,
                "anomaly_rate": 0.0
            }
    
    return stats


if __name__ == "__main__":
    # 测试跨设备数据集加载器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # 测试参数
    root_path = "data"
    
    # 测试获取客户端信息
    print("Testing client information...")
    clients = get_cross_device_client_ids(root_path)
    for client in clients:
        print(f"Client: {client['client_id']}, Device Type: {client['device_type']}, Source: {client['data_source']}")
    
    # 测试获取统计信息
    print("\nTesting device type statistics...")
    stats = get_device_type_statistics(root_path)
    for client_id, stat in stats.items():
        print(f"Client {client_id}: {stat['normal_samples']} normal, {stat['abnormal_samples']} abnormal, "
              f"rate: {stat['anomaly_rate']:.3f}")
    
    # 测试数据加载器
    if clients:
        client_id = clients[0]["client_id"]
        print(f"\nTesting dataloader for client {client_id}...")
        
        # 测试训练数据加载器
        train_loader = create_cross_device_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=True,
            batch_size=4,
            num_workers=0
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        for i, (spectrogram, label, device_type) in enumerate(train_loader):
            print(f"Batch {i}: spectrogram shape={spectrogram.shape}, labels={label}, device_type={device_type}")
            if i >= 2:  # 只测试前3个batch
                break
        
        # 测试测试数据加载器
        test_loader = create_cross_device_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=False,
            batch_size=4,
            num_workers=0
        )
        
        print(f"Test samples: {len(test_loader.dataset)}")
        for i, (spectrogram, label, device_type) in enumerate(test_loader):
            print(f"Batch {i}: spectrogram shape={spectrogram.shape}, labels={label}, device_type={device_type}")
            if i >= 2:  # 只测试前3个batch
                break
    
    print("\nAll tests passed!")
