# Fed-ProFiLA-AD: Cross-Device Technical Specification
## Federated Prototype-FiLMed Local Adapters for Cross-Device Acoustic Anomaly Detection

### 1. Application Background

**1.1. Context: Cross-Device Industrial Acoustic Anomaly Detection (CD-AAD)**
The primary application is cross-device acoustic anomaly detection in industrial settings, specifically for monitoring diverse machinery types (fans, valves, pumps) across different factories or production lines. This extends the original Fed-ProFiLA-AD to handle **cross-device heterogeneity** where different device types have fundamentally different acoustic characteristics.

**1.2. The Cross-Device Federated Challenge (CDFAAD)**
In real-world industrial scenarios, data is not only siloed by location but also by device type. Different factories may operate different types of machinery, and even within the same factory, there may be multiple device types. This necessitates a **Cross-Device Federated Learning** approach that can handle both data heterogeneity and device type heterogeneity.

**1.3. Core Problems in CDFAAD**
This project addresses three fundamental challenges:
1. **Non-IID Data:** Different devices have different "normal" acoustic patterns
2. **Device Type Heterogeneity:** Different device types (fans vs. valves) have fundamentally different acoustic characteristics
3. **Cross-Device Generalization:** Models trained on one device type should benefit from knowledge learned on other device types

**1.4. Proposed Solution: Cross-Device Fed-ProFiLA-AD**
We extend **Fed-ProFiLA-AD** to **Cross-Device Fed-ProFiLA-AD** by incorporating device type awareness and adaptive prototype alignment mechanisms.

---

### 2. Innovation Details (Formulation)

**2.1. Extended Nomenclature**
| Symbol | Definition | Scope |
| :--- | :--- | :--- |
| $i$ | Client index, $i \in \{1, \dots, N\}$ | - |
| $t$ | Communication round, $t \in \{0, \dots, T-1\}$ | - |
| $D_i$ | Local dataset for client $i$ (contains only `Normal` samples $x$) | Client |
| $T_i$ | Device type for client $i$ (e.g., "fan", "valve") | Client |
| $f_\theta$ | Shared Backbone Network (e.g., a CNN) | Global |
| $\theta$ | Parameters of the backbone, $\theta \in \mathbb{R}^P$ | Global (Aggregated) |
| $A_i$ | Local Adapter Network | Client (Personalized) |
| $\phi_i$ | Parameters of the adapter, $\phi_i \in \mathbb{R}^Q$. **Never uploaded.** | Client (Personalized) |
| $h$ | Device-Aware FiLM Parameter Generator Network | Global (Part of $\theta$) |
| $\mu_i$ | Local Prototype for client $i$ (center of normal features) | Client |
| $\bar{\mu}$ | Global Prototype (aggregated center of all normal features) | Global |
| $\mu_{T_i}$ | Device-Type Prototype for device type $T_i$ | Global |
| $z$ | Feature embedding, $z \in \mathbb{R}^d$ | - |
| $\lambda_{proto}$ | Hyperparameter: weight for prototype alignment loss | - |
| $\lambda_{device}$ | Hyperparameter: weight for device type alignment loss | - |
| $\alpha_{T_i}$ | Adaptive weight for device type $T_i$ | Global |

**2.2. Enhanced Model Forward Pass (The "Cross-Device ProFiLA" Mechanism)**
The core innovation extends the original ProFiLA mechanism to be device-type aware:

For an input spectrogram $x$ on client $i$ with device type $T_i$ during round $t$:
1. **Device-Aware FiLM Parameter Generation:** The network $h$ generates modulation parameters $(\gamma, \beta)$ from both the global prototype $\bar{\mu}_t$ and device type information $T_i$.
    $$
    (\gamma_t, \beta_t) = h(\bar{\mu}_t, T_i)
    $$
2. **Device-Conditioned Adaptation:** The local adapter $A_i$ processes the input $x$ with device type awareness.
    $$
    u = A_i(x; \phi_i, \gamma_t, \beta_t, T_i)
    $$
3. **Feature Extraction:** The shared backbone $f$ produces the final embedding $z$.
    $$
    z = f(u; \theta_i)
    $$

**2.3. Enhanced Prototype Definitions**
* **Local Prototype ($\mu_i$):** Same as original, but computed with device-aware model.
    $$
    \mu_i \leftarrow \frac{1}{|D_i|}\sum_{x \in D_i} f(A_i(x; \phi_i, \gamma_t, \beta_t, T_i); \theta_i)
    $$
* **Global Prototype ($\bar{\mu}$):** Weighted average considering device type importance.
    $$
    \bar{\mu}_{t+1} \leftarrow \sum_{i \in S_t} \alpha_{T_i} \cdot w_i \cdot \mu_i^{new}
    $$
* **Device-Type Prototype ($\mu_{T_i}$):** Prototype specific to device type $T_i$.
    $$
    \mu_{T_i} \leftarrow \frac{1}{|S_{T_i}|}\sum_{j \in S_{T_i}} \mu_j^{new}
    $$

**2.4. Enhanced Loss Functions (Local Objective)**
For a batch of embeddings $\{z_b\}$ from `Normal` samples on client $i$ with device type $T_i$:
1. **Task Loss (Compactness):** Pulls embeddings towards local prototype.
    $$
    \mathcal{L}_{task} = \mathbb{E}_{z \in \{z_b\}} [ \| z - \mu_i \|_2^2 ]
    $$
2. **Global Prototype Alignment Loss:** Aligns local prototype with global prototype.
    $$
    \mathcal{L}_{proto} = \| \mu_i - \bar{\mu}_t \|_2^2
    $$
3. **Device-Type Alignment Loss:** Aligns local prototype with device-type prototype.
    $$
    \mathcal{L}_{device} = \| \mu_i - \mu_{T_i} \|_2^2
    $$
4. **Total Loss:** Combined loss with adaptive weighting.
    $$
    \mathcal{L}_i = \mathcal{L}_{task} + \lambda_{proto} \cdot \mathcal{L}_{proto} + \lambda_{device} \cdot \mathcal{L}_{device}
    $$

---

### 3. Overall Algorithm Flow (Cross-Device Extension)

**Algorithm 2: Cross-Device Fed-ProFiLA-AD Training**

1. **Global Initialization:**
   * Server initializes global backbone parameters $\theta_0$, global prototype $\bar{\mu}_0 = \vec{0}$, and device-type prototypes $\{\mu_{T_k}\}_{k=1}^K = \vec{0}$.
   * Set global random seed to **42**.

2. **Input Data:** Audio files from multiple device types across different clients.

3. **Preprocessing:** Audio files are loaded, segmented, and converted to log-Mel spectrograms $x$ with device-type specific augmentation.

4. **Federated Training Loop (for $t = 0 \dots T-1$):**

   **A. Server Broadcast:**
   * Server selects a subset of clients $S_t$ (optionally balanced by device type).
   * Server broadcasts $\theta_t$, $\bar{\mu}_t$, and $\{\mu_{T_k}\}_{k=1}^K$ to all clients in $S_t$.

   **B. Client Local Update (Client $i \in S_t$ with device type $T_i$):**
   * i. **Receive & Set:** $\theta_i \leftarrow \theta_t$. Load local adapter parameters $\phi_i$.
   * ii. **Compute Device-Aware FiLM Params:** $(\gamma_t, \beta_t) \leftarrow h(\bar{\mu}_t, T_i)$.
   * iii. **Compute & Freeze $\mu_i$:** Calculate local prototype with device awareness.
   * iv. **Local Epochs (for $e = 1 \dots E$):**
       * For each batch $x_b$ from local $D_i$:
       * `# Forward Pass`
       * $u_b \leftarrow A_i(x_b; \phi_i, \gamma_t, \beta_t, T_i)$
       * $z_b \leftarrow f(u_b; \theta_i)$
       * `# Compute Loss`
       * $\mathcal{L}_i = \mathcal{L}_{task}(z_b, \mu_i) + \lambda_{proto} \cdot \mathcal{L}_{proto}(\mu_i, \bar{\mu}_t) + \lambda_{device} \cdot \mathcal{L}_{device}(\mu_i, \mu_{T_i})$
       * `# Backpropagation`
       * Compute gradients and update parameters.
   * v. **Compute New Prototype:** Calculate updated local prototype.
   * vi. **Upload:** Send $\theta_i$ and $\mu_i^{new}$ to server.

   **C. Server Aggregation:**
   * i. **Aggregate Backbone:** $\theta_{t+1} \leftarrow \text{FedAvg}(\{\theta_i\}_{i \in S_t})$
   * ii. **Aggregate Global Prototype:** $\bar{\mu}_{t+1} \leftarrow \text{WeightedAvg}(\{\mu_i^{new}\}_{i \in S_t})$
   * iii. **Aggregate Device-Type Prototypes:** $\mu_{T_k} \leftarrow \text{Avg}(\{\mu_i^{new}\}_{i \in S_t, T_i = T_k})$

5. **Training Output:** Final global model, prototypes, and personalized adapters.

6. **Inference Flow (Cross-Device Anomaly Detection):**
   * i. **Input:** Test spectrogram $x_{test}$ and device type $T_i$.
   * ii. **Load Personalized Model:** Client $i$ loads $\theta_T$, $\phi_i$, $\bar{\mu}_T$, $\mu_{T_i}$, and $\mu_i$.
   * iii. **Compute Device-Aware FiLM Params:** $(\gamma_T, \beta_T) \leftarrow h(\bar{\mu}_T, T_i)$.
   * iv. **Forward Pass:** $z_{test} \leftarrow f(A_i(x_{test}; \phi_i, \gamma_T, \beta_T, T_i); \theta_T)$.
   * v. **Output Anomaly Score:** 
       $$
       \text{Score} = \| z_{test} - \mu_i \|_2^2
       $$

---

### 4. Code Directory Structure (Cross-Device Extension)

```
fed_profila_ad/
    configs/
        cross_device_federation.yaml          # 跨设备联邦学习配置
    data/
        mimii_due_fan_0db/                    # DUE风扇数据（4个设备）
        mimii_dev_fan/                        # DEV风扇数据
        mimii_dev_valve/                      # DEV阀门数据
    dataset_loader/
        cross_device_dataset.py               # 跨设备数据集加载器
    models/
        device_aware_adapters.py              # 设备感知适配器
        backbone_cnn.py                       # 共享骨干网络
        adapters.py                           # 原始适配器
    methods/
        fed_profila_ad.py                    # 损失函数和工具函数
    trainers/
        server_loop.py                        # 服务器训练循环
        client_loop.py                        # 客户端训练循环
    eval/
        inference.py                          # 推理和评估
        metrics.py                            # 评估指标
    utils/
        seeding.py                            # 随机种子设置
        visualization.py                      # 可视化工具
    run_cross_device.py                       # 跨设备主运行脚本
    test_cross_device.py                     # 跨设备测试脚本
    dataset_loader/
```

---

### 5. Implementation Mapping (Cross-Device Innovation to Files)

| Algorithm Component / Innovation | Corresponding File(s) |
| :--- | :--- |
| **Cross-Device Data Partitioning** | `dataset_loader/cross_device_dataset.py` |
| **Device-Aware FiLM Generator** | `models/device_aware_adapters.py` |
| **Device-Conditioned Adapter** | `models/device_aware_adapters.py` |
| **Device-Type Prototype Management** | `trainers/server_loop.py` |
| **Enhanced Loss Functions** | `methods/fed_profila_ad.py` |
| **Cross-Device Client Selection** | `run_cross_device.py` |
| **Device-Type Statistics** | `dataset_loader/cross_device_dataset.py` |

---

### 6. Key Innovations for Cross-Device Learning

#### 6.1. Device Type Awareness
- **Device Type Encoding:** Each device type is encoded as a learnable embedding
- **Device-Specific Augmentation:** Different augmentation strategies for different device types
- **Device-Type Prototypes:** Separate prototypes for each device type

#### 6.2. Adaptive Prototype Alignment
- **Multi-Level Alignment:** Global, device-type, and local prototype alignment
- **Adaptive Weighting:** Dynamic adjustment of alignment weights based on device type
- **Cross-Device Knowledge Transfer:** Knowledge sharing across different device types

#### 6.3. Enhanced Client Selection
- **Device-Type Balanced Selection:** Ensure representation from all device types
- **Adaptive Selection:** Adjust selection strategy based on device type performance
- **Cross-Device Evaluation:** Evaluate both within-type and cross-type performance

---

### 7. Experimental Design

#### 7.1. Baseline Comparisons
1. **Single Device Training:** Each client trains independently
2. **Centralized Training:** All data trained together
3. **Standard Federated Learning:** FedAvg without device awareness
4. **Original Fed-ProFiLA-AD:** Without cross-device extensions
5. **Cross-Device Fed-ProFiLA-AD:** Our proposed method

#### 7.2. Cross-Device Evaluation
1. **Within-Type Performance:** Performance on same device type
2. **Cross-Type Performance:** Performance across different device types
3. **Generalization Analysis:** How well models generalize to new device types
4. **Knowledge Transfer Analysis:** What knowledge transfers between device types

#### 7.3. Ablation Studies
1. **Device Type Awareness:** Impact of device type information
2. **Multi-Level Prototype Alignment:** Contribution of each alignment level
3. **Adaptive Weighting:** Impact of adaptive weight adjustment
4. **Client Selection Strategy:** Impact of different selection methods

---

### 8. Expected Benefits

#### 8.1. Technical Benefits
- **Better Generalization:** Models learn from diverse device types
- **Improved Robustness:** More robust to device type variations
- **Knowledge Transfer:** Effective knowledge sharing across device types
- **Scalability:** Easy to add new device types

#### 8.2. Practical Benefits
- **Real-World Applicability:** Matches real industrial scenarios
- **Reduced Data Requirements:** Can leverage data from different device types
- **Flexible Deployment:** Can adapt to different industrial settings
- **Maintenance Efficiency:** Easier to maintain and update models

---

