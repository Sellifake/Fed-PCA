"""
Base Dataset Loader for Fed-ProFiLA-AD (基础版本)
实现基础的MIMII数据集加载器，支持单个设备类型
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


class MIMIIDataset(Dataset):
    """
    MIMII数据集类 (基础版本)
    支持MIMII-DUE风扇数据集的加载
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
        max_samples: int = None,  # 限制最大样本数（用于快速测试）
        include_abnormal_in_train: bool = True,
        abnormal_fraction: float = 0.2
    ):
        """
        初始化MIMII数据集
        
        Args:
            root_path: 数据集根目录路径
            client_id: 客户端ID (如 "id_00", "id_02" 等)
            is_train: 是否为训练模式
            sample_rate: 音频采样率
            segment_length: 音频片段长度
            n_mels: Mel频谱图频率bins数量
            hop_length: 频谱图hop长度
            n_fft: FFT窗口大小
        """
        self.root_path = root_path
        self.client_id = client_id
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.max_samples = max_samples
        self.include_abnormal_in_train = include_abnormal_in_train
        self.abnormal_fraction = abnormal_fraction
        
        # 构建文件路径
        self.client_path = self._get_client_path(root_path, client_id)
        
        # 初始化文件列表和标签
        self.file_paths = []
        self.labels = []
        
        # 加载数据
        self._load_data()
        
        # 限制数据量（用于快速测试）
        if self.max_samples is not None and len(self.file_paths) > self.max_samples:
            import random
            indices = list(range(len(self.file_paths)))
            random.seed(42)  # 固定随机种子
            selected_indices = random.sample(indices, self.max_samples)
            self.file_paths = [self.file_paths[i] for i in selected_indices]
            self.labels = [self.labels[i] for i in selected_indices]
        
        # 初始化音频变换
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # 计算客户端CMVN统计，稳定特征分布
        self.cmvn_mean = None
        self.cmvn_std = None
        try:
            self._compute_cmvn_stats(max_files=256)
            if self.cmvn_mean is not None:
                logger.info(
                    f"Client {self.client_id} CMVN ready: mean={float(self.cmvn_mean.mean()):.4f}, std={float(self.cmvn_std.mean()):.4f}"
                )
        except Exception as e:
            logger.warning(f"CMVN stats computation failed for {self.client_id}: {e}")
    
    def _get_client_path(self, root_path: str, client_id: str) -> str:
        """获取客户端数据路径"""
        if client_id.startswith('id_'):
            return os.path.join(root_path, "mimii_due_fan_0db", client_id)
        elif client_id == "dev_fan":
            return os.path.join(root_path, "mimii_dev_fan")
        elif client_id == "dev_valve":
            return os.path.join(root_path, "mimii_dev_valve")
        else:
            raise ValueError(f"Unknown client_id: {client_id}")
    
    def _load_data(self):
        """加载数据文件"""
        if not os.path.exists(self.client_path):
            logger.warning(f"Client path does not exist: {self.client_path}")
            return
        
        if self.client_id.startswith('id_'):
            self._load_due_data()
        else:
            self._load_dev_data()
    
    def _load_due_data(self):
        """加载MIMII-DUE数据集（按normal/abnormal组织）"""
        if self.is_train:
            normal_path = os.path.join(self.client_path, "normal")
            if os.path.exists(normal_path):
                normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
                self.file_paths.extend(normal_files)
                self.labels.extend([0] * len(normal_files))

            if self.include_abnormal_in_train:
                abnormal_path = os.path.join(self.client_path, "abnormal")
                if os.path.exists(abnormal_path):
                    abnormal_files = glob.glob(os.path.join(abnormal_path, "*.wav"))
                    import random
                    random.seed(42)
                    k = int(len(self.file_paths) * self.abnormal_fraction)
                    k = max(1, min(k, len(abnormal_files))) if abnormal_files else 0
                    if k > 0:
                        sampled_abn = random.sample(abnormal_files, k)
                        self.file_paths.extend(sampled_abn)
                        self.labels.extend([1] * len(sampled_abn))
            # 打乱
            if len(self.file_paths) > 0:
                combined = list(zip(self.file_paths, self.labels))
                import random
                random.seed(42)
                random.shuffle(combined)
                self.file_paths, self.labels = zip(*combined)
                self.file_paths, self.labels = list(self.file_paths), list(self.labels)
        else:
            # 测试时包含Normal和Abnormal数据
            normal_path = os.path.join(self.client_path, "normal")
            abnormal_path = os.path.join(self.client_path, "abnormal")
            
            if os.path.exists(normal_path):
                normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
                self.file_paths.extend(normal_files)
                self.labels.extend([0] * len(normal_files))
            
            if os.path.exists(abnormal_path):
                abnormal_files = glob.glob(os.path.join(abnormal_path, "*.wav"))
                self.file_paths.extend(abnormal_files)
                self.labels.extend([1] * len(abnormal_files))
    
    def _load_dev_data(self):
        """加载MIMII-DEV数据集（按train/test组织）"""
        if self.is_train:
            normal_path = os.path.join(self.client_path, "train", "normal")
            if os.path.exists(normal_path):
                normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
                self.file_paths.extend(normal_files)
                self.labels.extend([0] * len(normal_files))
            
            if self.include_abnormal_in_train:
                abnormal_path = os.path.join(self.client_path, "train", "abnormal")
                if os.path.exists(abnormal_path):
                    abnormal_files = glob.glob(os.path.join(abnormal_path, "*.wav"))
                    import random
                    random.seed(42)
                    k = int(len(self.file_paths) * self.abnormal_fraction)
                    k = max(1, min(k, len(abnormal_files))) if abnormal_files else 0
                    if k > 0:
                        sampled_abn = random.sample(abnormal_files, k)
                        self.file_paths.extend(sampled_abn)
                        self.labels.extend([1] * len(sampled_abn))
            
            if len(self.file_paths) > 0:
                combined = list(zip(self.file_paths, self.labels))
                import random
                random.seed(42)
                random.shuffle(combined)
                self.file_paths, self.labels = zip(*combined)
                self.file_paths, self.labels = list(self.file_paths), list(self.labels)
        else:
            # 测试时包含Normal和Abnormal数据
            normal_path = os.path.join(self.client_path, "test", "normal")
            abnormal_path = os.path.join(self.client_path, "test", "abnormal")
            
            if os.path.exists(normal_path):
                normal_files = glob.glob(os.path.join(normal_path, "*.wav"))
                self.file_paths.extend(normal_files)
                self.labels.extend([0] * len(normal_files))
            
            if os.path.exists(abnormal_path):
                abnormal_files = glob.glob(os.path.join(abnormal_path, "*.wav"))
                self.file_paths.extend(abnormal_files)
                self.labels.extend([1] * len(abnormal_files))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        获取数据样本
        
        Returns:
            Tuple[torch.Tensor, int]: (频谱图, 标签)
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # 加载音频文件
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            # 返回空音频
            waveform = torch.zeros(1, self.segment_length)
            sr = self.sample_rate
        
        # 处理多通道音频：如果有多通道，转换为单声道（取平均或第一通道）
        if waveform.shape[0] > 1:
            # 方法1：取平均（推荐）
            waveform = waveform.mean(dim=0, keepdim=True)
            # 方法2：或者只取第一通道
            # waveform = waveform[0:1, :]
        
        # 调整采样率
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        # 确保音频长度为segment_length
        if waveform.shape[1] < self.segment_length:
            # 填充
            padding = self.segment_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif waveform.shape[1] > self.segment_length:
            # 裁剪
            waveform = waveform[:, :self.segment_length]
        
        # 转换为Mel频谱图
        # waveform形状: [1, segment_length]
        # mel_spec形状: [1, n_mels, time_frames]
        mel_spec = self.mel_transform(waveform)
        
        # 转换为对数尺度 (log-mel spectrogram)
        mel_spec = torch.log(mel_spec + 1e-6)
        
        # Client-level CMVN 标准化（优先使用），否则退化为保守映射
        if self.cmvn_mean is not None and self.cmvn_std is not None:
            # 广播到 [1, n_mels, time] 形状
            mel_spec = (mel_spec - self.cmvn_mean) / (self.cmvn_std + 1e-6)
            # 将标准化后的值裁剪并映射到[0,1]，便于网络稳定训练
            mel_spec = torch.clamp(mel_spec, -3.0, 3.0)
            mel_spec = (mel_spec + 3.0) / 6.0
        else:
            # 保守映射：适配不同数据域
            mel_spec = (mel_spec + 5.0) / 10.0
            mel_spec = torch.clamp(mel_spec, 0.0, 1.0)
        
        # 确保输出形状为 [1, n_mels, time_frames]
        # mel_spec已经是 [1, n_mels, time_frames]，无需额外处理
        
        return mel_spec, label

    def _compute_cmvn_stats(self, max_files: int = 256) -> None:
        """计算客户端级别的CMVN统计（Mel维度上的均值与方差）。"""
        if len(self.file_paths) == 0:
            return
        import random
        indices = list(range(len(self.file_paths)))
        random.seed(42)
        random.shuffle(indices)
        indices = indices[:min(max_files, len(indices))]
        sum_vec = None
        sum_sq_vec = None
        count_frames = 0
        for i in indices:
            file_path = self.file_paths[i]
            try:
                waveform, sr = torchaudio.load(file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    resampler = T.Resample(sr, self.sample_rate)
                    waveform = resampler(waveform)
                mel = self.mel_transform(waveform)  # [1, n_mels, T]
                mel = torch.log(mel + 1e-6)
                # 沿时间维累计
                mel = mel.squeeze(0)  # [n_mels, T]
                if sum_vec is None:
                    sum_vec = mel.sum(dim=1)  # [n_mels]
                    sum_sq_vec = (mel ** 2).sum(dim=1)
                else:
                    sum_vec += mel.sum(dim=1)
                    sum_sq_vec += (mel ** 2).sum(dim=1)
                count_frames += mel.shape[1]
            except Exception:
                continue
        if count_frames > 0:
            mean = (sum_vec / count_frames).view(1, -1, 1)  # [1, n_mels, 1]
            var = (sum_sq_vec / count_frames) - (sum_vec / count_frames) ** 2
            std = torch.sqrt(torch.clamp(var, min=1e-6)).view(1, -1, 1)
            self.cmvn_mean = mean
            self.cmvn_std = std


def create_dataloader(
    root_path: str,
    client_id: str,
    is_train: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
    sample_rate: int = 16000,
    segment_length: int = 4096,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 1024,
    max_samples: int = None,  # 限制最大样本数（用于快速测试）
    include_abnormal_in_train: bool = True,
    abnormal_fraction: float = 0.2,
    balanced_sampling: bool = True,
    abnormal_ratio_per_batch: float = 0.25
) -> Optional[DataLoader]:
    """
    创建数据加载器
    
    Args:
        root_path: 数据集根目录
        client_id: 客户端ID
        is_train: 是否为训练模式
        batch_size: 批次大小
        num_workers: 工作进程数
        pin_memory: 是否使用pin_memory
        其他参数: 音频预处理参数
        
    Returns:
        Optional[DataLoader]: 数据加载器
    """
    try:
        dataset = MIMIIDataset(
            root_path=root_path,
            client_id=client_id,
            is_train=is_train,
            sample_rate=sample_rate,
            segment_length=segment_length,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            max_samples=max_samples,
            include_abnormal_in_train=include_abnormal_in_train,
            abnormal_fraction=abnormal_fraction
        )
        
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for client {client_id}")
            return None
        
        if is_train and balanced_sampling and hasattr(dataset, 'labels') and len(set(dataset.labels)) > 1:
            sampler = BalancedBatchSampler(dataset.labels, batch_size, abnormal_ratio_per_batch)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=is_train,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=is_train  # 训练时drop_last
            )
        
        return dataloader
        
    except Exception as e:
        logger.error(f"Error creating dataloader for client {client_id}: {e}")
        return None


def get_client_ids(root_path: str) -> List[str]:
    """
    获取所有客户端ID（自动检测MIMII-DUE和MIMII-DEV数据集）
    
    Args:
        root_path: 数据集根目录
        
    Returns:
        List[str]: 客户端ID列表
    """
    client_ids = []
    
    # 检测MIMII-DUE数据集
    due_fan_path = os.path.join(root_path, "mimii_due_fan_0db")
    if os.path.exists(due_fan_path):
        due_clients = [
            d for d in os.listdir(due_fan_path)
            if os.path.isdir(os.path.join(due_fan_path, d)) and d.startswith('id_')
        ]
        client_ids.extend(sorted(due_clients))
    
    # 检测MIMII-DEV Fan数据集
    dev_fan_path = os.path.join(root_path, "mimii_dev_fan")
    if os.path.exists(dev_fan_path):
        client_ids.append("dev_fan")
    
    # 检测MIMII-DEV Valve数据集
    dev_valve_path = os.path.join(root_path, "mimii_dev_valve")
    if os.path.exists(dev_valve_path):
        client_ids.append("dev_valve")
    
    if not client_ids:
        logger.warning(f"No clients found in {root_path}")
    
    return sorted(client_ids)


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    简单的类别均衡采样器：按给定比例在每个batch中采样异常样本，其余补正常样本。
    """
    def __init__(self, labels, batch_size: int, abnormal_ratio: float = 0.25):
        self.labels = list(labels)
        self.batch_size = batch_size
        self.abn_ratio = float(min(max(abnormal_ratio, 0.0), 0.5))
        self.indices_norm = [i for i, y in enumerate(self.labels) if y == 0]
        self.indices_abn = [i for i, y in enumerate(self.labels) if y == 1]
        self.num_abn_per_batch = max(1, int(round(self.batch_size * self.abn_ratio)))
        self.num_norm_per_batch = self.batch_size - self.num_abn_per_batch
        import random
        self._rng = random.Random(42)

    def __iter__(self):
        import random
        rng = self._rng
        norm = self.indices_norm.copy()
        abn = self.indices_abn.copy()
        rng.shuffle(norm)
        rng.shuffle(abn)
        norm_ptr, abn_ptr = 0, 0
        batches = []
        while norm_ptr + self.num_norm_per_batch <= len(norm) and abn_ptr + self.num_abn_per_batch <= len(abn):
            batch = norm[norm_ptr:norm_ptr + self.num_norm_per_batch] + abn[abn_ptr:abn_ptr + self.num_abn_per_batch]
            rng.shuffle(batch)
            batches.append(batch)
            norm_ptr += self.num_norm_per_batch
            abn_ptr += self.num_abn_per_batch
        # 打乱所有batch顺序
        rng.shuffle(batches)
        for b in batches:
            for idx in b:
                yield idx

    def __len__(self):
        # 返回能整除的样本数
        num_batches = min(
            len(self.indices_norm) // self.num_norm_per_batch,
            len(self.indices_abn) // self.num_abn_per_batch
        )
        return num_batches * self.batch_size


if __name__ == "__main__":
    # 测试数据集加载器
    import logging
    logging.basicConfig(level=logging.INFO)
    
    root_path = "data"
    client_ids = get_client_ids(root_path)
    print(f"Found clients: {client_ids}")
    
    if client_ids:
        client_id = client_ids[0]
        print(f"\nTesting with client: {client_id}")
        
        # 测试训练数据加载器
        train_loader = create_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=True,
            batch_size=4
        )
        
        if train_loader:
            print(f"Train loader created: {len(train_loader.dataset)} samples")
            for batch_x, batch_y in train_loader:
                print(f"Batch shape: {batch_x.shape}, Labels: {batch_y}")
                break
        
        # 测试测试数据加载器
        test_loader = create_dataloader(
            root_path=root_path,
            client_id=client_id,
            is_train=False,
            batch_size=4
        )
        
        if test_loader:
            print(f"\nTest loader created: {len(test_loader.dataset)} samples")
            for batch_x, batch_y in test_loader:
                print(f"Batch shape: {batch_x.shape}, Labels: {batch_y}")
                break

