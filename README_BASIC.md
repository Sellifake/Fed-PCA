# Fed-ProFiLA-AD 基础版本

这是一个简化后的基础版本，移除了所有跨设备相关的创新，只保留了核心的Fed-ProFiLA-AD功能。

## 核心组件

### 1. 模型架构

- **BackboneCNN**: 共享骨干网络
  - FiLM生成器：从全局原型生成FiLM参数 (γ, β)
  - 本地适配器：使用FiLM参数进行条件调制
  - 主编码器：提取特征嵌入

### 2. 训练流程

1. **服务器初始化**
   - 初始化全局模型参数 θ₀
   - 初始化全局原型 μ̄₀

2. **客户端本地训练**
   - 接收全局参数和原型
   - 计算本地原型 μᵢ
   - 使用损失函数训练：L = L_task + λ_proto × L_proto

3. **服务器聚合**
   - FedAvg聚合模型参数
   - 加权平均聚合原型

## 使用方法

### 快速开始

```bash
# 使用默认配置运行
python run_basic.py

# 指定配置文件
python run_basic.py --config configs/basic_federation.yaml

# 指定设备
python run_basic.py --device cuda
```

### 配置文件说明

基础配置文件位于 `configs/basic_federation.yaml`：

- **dataset**: 数据集配置（路径、音频参数）
- **model**: 模型配置（特征维度、dropout等）
- **training**: 训练配置（batch size、学习率等）
- **federation**: 联邦学习配置（轮数、epochs、lambda_proto）
- **system**: 系统配置（设备、种子等）

## 代码结构

```
├── models/
│   ├── backbone_cnn.py      # 基础骨干网络（简化版）
│   └── adapters.py           # FiLM生成器和适配器
├── methods/
│   └── fed_profila_ad.py    # 损失函数和工具函数（简化版）
├── dataset_loader/
│   └── base_dataset.py       # 基础数据集加载器
├── trainers/
│   ├── server_loop.py       # 服务器训练循环
│   └── client_loop.py       # 客户端训练循环（简化版）
├── eval/
│   ├── inference.py         # 推理和评估
│   └── metrics.py           # 评估指标
├── run_basic.py             # 基础运行脚本
└── configs/
    └── basic_federation.yaml # 基础配置文件
```

## 主要简化内容

与跨设备版本相比，基础版本移除了：

1. ❌ 设备类型编码器
2. ❌ 设备类型感知的FiLM生成
3. ❌ 设备类型原型
4. ❌ 设备类型对齐损失
5. ❌ 跨设备数据集支持（仅支持MIMII-DUE风扇数据）

## 核心算法流程

```
For round t = 0 to T-1:
  1. Server广播:
     - θ_t (全局模型参数)
     - μ̄_t (全局原型)
  
  2. Client本地更新 (对每个客户端 i):
     - 接收 θ_t, μ̄_t
     - 计算本地原型 μ_i
     - 本地训练 E 轮:
       * Forward: z = f(A_i(x; γ_t, β_t); θ_t)
       * Loss: L = ||z - μ_i||² + λ_proto × ||μ_i - μ̄_t||²
       * Backward: 更新参数
     - 上传 θ_i^{new}, μ_i^{new}
  
  3. Server聚合:
     - θ_{t+1} = FedAvg({θ_i^{new}})
     - μ̄_{t+1} = WeightedAvg({μ_i^{new}})
```

## 注意事项

1. **数据集路径**: 确保 `data/mimii_due_fan_0db/` 目录存在，且包含客户端子目录（如 `id_00`, `id_02` 等）

2. **设备类型**: 基础版本仅支持单一设备类型（风扇），不支持跨设备类型训练

3. **配置**: 如果配置文件不存在，程序会自动创建一个默认配置文件

4. **日志**: 训练日志保存在 `logs/fed_profila_ad.log`

5. **输出**: 模型和结果保存在 `outputs/run_YYYYMMDD_HHMMSS/` 目录

## 下一步

在基础版本运行正常后，可以逐步添加：
- 跨设备类型支持
- 设备类型感知机制
- 更复杂的损失函数
- 其他创新功能

