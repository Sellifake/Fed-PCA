# Fed-ProFiLA-AD (Cross-Device Edition)

**Federated Prototype-FiLMed Local Adapters for Cross-Device Acoustic Anomaly Detection**

联邦学习跨设备声学异常检测：基于原型条件局部适配器的方法

## 项目概述

Fed-ProFiLA-AD 是一个创新的联邦学习框架，专门用于工业声学异常检测。该方法解决了跨设备联邦学习中的非独立同分布（Non-IID）数据问题和设备类型异构性问题，通过结合设备感知机制、共享全局骨干网络、轻量级本地适配器和基于原型的条件机制来实现跨设备个性化联邦学习。

### 跨设备版本特性

本版本专门针对**跨设备联邦学习**场景设计，支持：
- **多设备类型**：Fan（风扇）、Valve（阀门）等多种工业设备
- **设备类型感知**：自适应处理不同设备类型的声学特征差异
- **多级原型对齐**：全局、设备类型、本地三级知识共享
- **跨设备泛化**：有效处理设备类型间的知识迁移

## 核心创新

1. **设备类型感知机制**：使用设备类型编码和自适应FiLM参数生成
2. **原型条件机制**：使用全局原型和设备类型信息生成FiLM参数
3. **个性化适配器**：每个客户端维护轻量级本地适配器，永不共享
4. **多级损失函数**：任务损失 + 全局原型对齐 + 设备类型对齐
5. **联邦聚合**：同时聚合模型参数、全局原型和设备类型原型

## 项目结构

```
fed_profila_ad/
├── configs/
│   ├── cross_device_federation.yaml      # 跨设备联邦学习配置（主配置）
│   └── mimii_due.yaml                    # MIMII-DUE配置（旧配置）
├── data/
│   ├── mimii_due_fan_0db/               # DUE风扇数据（4个设备）
│   ├── mimii_dev_fan/                   # DEV风扇数据
│   └── mimii_dev_valve/                  # DEV阀门数据
├── dataset_loader/
│   ├── cross_device_dataset.py          # 跨设备数据集加载器（推荐使用）
│   └── dataset_mimii.py                 # 原始数据集加载器（已弃用）
├── models/
│   ├── device_aware_adapters.py         # 设备感知适配器（核心模块）
│   ├── backbone_cnn.py                  # 共享骨干网络
│   └── adapters.py                      # 原始适配器
├── methods/
│   └── fed_profila_ad.py                # 损失函数和工具函数
├── trainers/
│   ├── server_loop.py                   # 服务器训练循环
│   └── client_loop.py                   # 客户端训练循环
├── eval/
│   ├── inference.py                     # 推理和评估
│   └── metrics.py                       # 评估指标
├── utils/
│   ├── seeding.py                       # 随机种子设置
│   └── visualization.py                 # 可视化工具
├── run_cross_device.py                   # 跨设备主运行脚本（推荐使用）
├── run.py                                # 原始运行脚本
├── test_cross_device.py                 # 跨设备测试脚本
├── 数据集划分方案_跨设备版.md            # 数据集划分文档
├── fedpca_innovation_跨设备版.md         # 技术规范文档
└── README.md                             # 本文档
```

## 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd fed_profila_ad

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保MIMII数据集已正确放置在`data/`目录下：

```
data/
├── mimii_due_fan_0db/
│   ├── id_00/
│   │   ├── normal/
│   │   └── abnormal/
│   ├── id_02/
│   │   ├── normal/
│   │   └── abnormal/
│   ├── id_04/
│   │   ├── normal/
│   │   └── abnormal/
│   └── id_06/
│       ├── normal/
│       └── abnormal/
├── mimii_dev_fan/
│   ├── train/normal/
│   └── test/abnormal/
└── mimii_dev_valve/
    ├── train/normal/
    └── test/abnormal/
```

### 3. 运行跨设备联邦学习

```bash
# 使用跨设备配置运行（推荐）
python run_cross_device.py --config configs/cross_device_federation.yaml

# 或使用默认配置
python run_cross_device.py
```

### 4. 测试实现

```bash
# 运行跨设备测试
python test_cross_device.py
```

## 数据集

项目使用MIMII（Machine Incident Dataset in Industrial Settings）数据集：

### 数据统计

| 客户端ID | 设备类型 | Normal样本 | Abnormal样本 | 总样本数 | 异常率 | 数据来源 |
|---------|---------|------------|--------------|---------|--------|---------|
| id_00   | Fan     | 1,011      | 407          | 1,418   | 28.7%  | MIMII-DUE |
| id_02   | Fan     | 1,016      | 359          | 1,375   | 26.1%  | MIMII-DUE |
| id_04   | Fan     | 1,033      | 348          | 1,381   | 25.2%  | MIMII-DUE |
| id_06   | Fan     | 1,016      | 361          | 1,377   | 26.2%  | MIMII-DUE |
| dev_fan | Fan     | 3,000      | 600          | 3,600   | 16.7%  | MIMII-DEV |
| dev_valve | Valve | 3,000      | 600          | 3,600   | 16.7%  | MIMII-DEV |
| **总计** | **-** | **10,076** | **2,675**    | **12,751** | **21.0%** | **-** |

### 数据集结构

#### MIMII-DUE 数据集
- **位置**: `data/mimii_due_fan_0db/`
- **设备类型**: 风扇 (Fan)
- **客户端数量**: 4个 (id_00, id_02, id_04, id_06)
- **数据组织**: 每个客户端包含 `normal/` 和 `abnormal/` 子目录
- **文件格式**: WAV音频文件，采样率16kHz

#### MIMII-DEV 数据集
- **位置**: `data/mimii_dev_fan/` 和 `data/mimii_dev_valve/`
- **设备类型**: 风扇 (Fan) 和 阀门 (Valve)
- **数据组织**: 按 `train/` 和 `test/` 划分
  - `train/` 目录：所有文件都是 normal（训练用）
  - `test/` 目录：所有文件都是 abnormal（测试用）
- **文件格式**: WAV音频文件，采样率16kHz

### 客户端配置

- **6个客户端**：4个DUE Fan + 1个DEV Fan + 1个DEV Valve
- **设备类型多样性**：风扇和阀门两种工业设备
- **数据异构性**：不同设备的异常模式和数据分布不同

## 配置说明

### 主要配置文件

- `configs/cross_device_federation.yaml`：跨设备联邦学习主配置（推荐使用）
- `configs/mimii_due.yaml`：MIMII-DUE配置（已弃用）

### 关键配置参数

#### 数据集配置
```yaml
dataset:
  root_path: "data"  # 数据集根目录
  client_mapping:    # 客户端映射
    id_00: {device_type: "fan", dataset: "due"}
    id_02: {device_type: "fan", dataset: "due"}
    id_04: {device_type: "fan", dataset: "due"}
    id_06: {device_type: "fan", dataset: "due"}
    dev_fan: {device_type: "fan", dataset: "dev"}
    dev_valve: {device_type: "valve", dataset: "dev"}
```

#### 联邦学习配置
```yaml
federation:
  num_clients: 6                                    # 客户端数量
  client_selection: "all"                          # 客户端选择策略
  num_rounds: 100                                  # 通信轮数
  local_epochs: 5                                 # 本地训练轮数
  
  cross_device:
    enable_prototype_alignment: true               # 启用原型对齐
    lambda_proto: 0.1                              # 原型对齐权重
    device_type_aware: true                        # 设备类型感知
    adaptive_lambda: true                          # 自适应权重
```

#### 模型配置
```yaml
model:
  device_aware:
    enabled: true                                  # 启用设备感知
    device_type_dim: 16                            # 设备类型特征维度
    adaptive_lambda: true                          # 自适应权重调整
```

#### 音频预处理配置
```yaml
audio:
  sample_rate: 16000
  n_mels: 128
  n_fft: 1024
  hop_length: 512
  max_length: 64000
  normalize: true
```

## 核心算法

### 跨设备Fed-ProFiLA-AD流程

1. **初始化**：服务器初始化全局模型和原型
2. **客户端选择**：按设备类型平衡选择客户端
3. **设备感知训练**：
   - 生成设备感知的FiLM参数
   - 使用本地适配器进行个性化训练
   - 计算多级损失（任务损失 + 全局对齐 + 设备类型对齐）
4. **多级聚合**：
   - 聚合模型参数（FedAvg）
   - 聚合全局原型
   - 聚合设备类型原型
5. **评估**：在测试集上评估跨设备性能

### 数学公式

**设备感知FiLM生成**：
$$(\gamma_t, \beta_t) = h(\bar{\mu}_t, T_i)$$

**特征提取**：
$$u = A_i(x; \phi_i, \gamma_t, \beta_t, T_i)$$
$$z = f(u; \theta_i)$$

**多级损失**：
$$\mathcal{L}_i = \mathcal{L}_{task} + \lambda_{proto} \cdot \mathcal{L}_{proto} + \lambda_{device} \cdot \mathcal{L}_{device}$$

## 实验结果

### 预期性能

- **跨设备泛化**：AUC > 0.85，F1 > 0.80
- **设备内性能**：AUC > 0.90，F1 > 0.85
- **收敛速度**：50-100轮内收敛

### 评估指标

- **异常检测性能**：AUC、F1-Score、Precision、Recall
- **联邦学习性能**：收敛速度、通信效率、个性化程度
- **跨设备性能**：设备内/设备间性能、类型内/类型间泛化

## 文档

- **技术规范**：`fedpca_innovation_跨设备版.md` - 详细的算法设计和实现
- **数据集划分**：`数据集划分方案_跨设备版.md` - 完整的数据集划分方案

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License

## 引用

如果您使用本项目，请引用：

```bibtex
@article{fedprofila_ad,
  title={Fed-ProFiLA-AD: Federated Prototype-FiLMed Local Adapters for Cross-Device Acoustic Anomaly Detection},
  author={...},
  journal={...},
  year={2024}
}
```

## 联系方式

如有问题，请提交Issue或联系项目维护者。
