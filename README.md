# hyformer

## 基于 naive 分支的两条“最佳方案”

> 目标 1：去掉 `Loss/train` 尖刺（优先保证训练稳定性与可复现）。  
> 目标 2：加入 ESMM（CTR + CTCVR 多任务，提升 CVR 学习质量）。

---

## 方案 A（优先落地）：去掉 `Loss/train` 尖刺

结合当前代码，`trainer.py` 已经有 `clip_grad_norm_(max_norm=1.0)`，但仍可能出现尖刺，最常见原因是：

- **稀疏参数学习率过大**（当前默认 `sparse_lr=0.05`）导致 embedding 更新突变；
- **dense 学习率无 warmup**，前几千 step 梯度分布尚未稳定；
- **日志粒度是 step 级 raw loss**，天然噪声较大，视觉上“尖刺”。

### A1. 稀疏/稠密优化器分级降速（最有效）

建议从 naive baseline 起步参数：

- `--lr 5e-5`（dense，先减半）
- `--sparse_lr 0.01`（sparse，从 0.05 下调到 0.01）
- `--batch_size` 尽量增大（显存允许时优先 512/1024）

### A2. 引入 warmup + cosine decay（dense 优化器）

推荐：

- warmup 比例 `3%~5%` steps；
- warmup 后使用 cosine 衰减到 `lr * 0.1`。

这样可以显著降低训练前期的 loss 突发峰值。

### A3. 全局梯度裁剪改为“可配置 + 日志化”

建议把梯度裁剪阈值改为可配置（如 `--grad_clip_norm 0.5~1.0`），并将每步 grad norm 写入 tensorboard：

- 先看分布再调阈值，不建议盲调。

### A4. 训练日志从 raw loss 改为 EMA loss

保留原始 `Loss/train_raw`，新增：

- `Loss/train_ema`（如 `ema=0.95`）

这一步不改变训练，只改变可观测性，能去除视觉尖刺噪声，便于对比实验。

### A5. 稳定性最小实验矩阵（建议）

- Exp-S1: `sparse_lr=0.05`（现状）
- Exp-S2: `sparse_lr=0.01`
- Exp-S3: `sparse_lr=0.01 + warmup/cosine`
- Exp-S4: `S3 + grad_clip_norm=0.5`

以 **valid AUC + valid logloss + train_ema 波动率** 选最优。

---

## 方案 B（结构升级）：加入 ESMM（推荐正式版本）

### B1. ESMM 的最小正确建模

ESMM 需要两个塔输出：

- `pCTR = P(click=1 | x)`
- `pCVR = P(convert=1 | click=1, x)`
- `pCTCVR = pCTR * pCVR`

训练目标：

- CTR 头监督 `click_label`
- CTCVR 监督 `convert_label`（全样本监督，未点击样本 `convert_label=0`）

> 关键点：不要直接用“仅点击样本上的 CVR loss”作为主监督，否则会回到 sample selection bias。

### B2. 数据侧改造（先做）

在数据集输出中新增两个 label：

- `click_label`（是否点击）
- `convert_label`（是否转化）

并保持原 `label` 兼容旧逻辑（例如等于 `convert_label`）。

### B3. 模型侧改造（主干共享 + 双头）

保留当前 HyFormer backbone，末端改为：

- `ctr_head: Linear(D, 1)`
- `cvr_head: Linear(D, 1)`

前向输出：

- `ctr_logit`
- `cvr_logit`
- `ctcvr_prob = sigmoid(ctr_logit) * sigmoid(cvr_logit)`

### B4. 损失函数（推荐权重）

- `loss_ctr = BCEWithLogits(ctr_logit, click_label)`
- `loss_ctcvr = BCE(ctcvr_prob, convert_label)`
- `loss_total = w_ctr * loss_ctr + w_ctcvr * loss_ctcvr`

推荐初值：

- `w_ctr=1.0`
- `w_ctcvr=1.0`（若 convert 极稀疏可提高到 1.5~2.0）

### B5. 指标与推理

训练/评估至少记录：

- CTR-AUC（click_label vs pCTR）
- CTCVR-AUC（convert_label vs pCTCVR）
- 主监控建议仍使用 CTCVR-AUC + CTCVR-logloss

线上若要预估 CVR：

- 直接用 `pCVR`（条件转化率）
- 若要预估最终转化概率，用 `pCTCVR`

---

## 推荐落地顺序（最稳）

1. 先完成方案 A（训练稳定化），拿到稳定 baseline；
2. 在稳定 baseline 上接入方案 B（ESMM）；
3. 最终比较 `naive vs stabilized vs stabilized+ESMM` 三组结果。

---

## 一句话结论

- **去尖刺最佳方案**：`sparse_lr 下调 + warmup/cosine + 可配置梯度裁剪 + EMA 监控`。  
- **加 ESMM最佳方案**：`共享主干双头（CTR/CVR）+ 以 CTCVR 为核心监督`，并保证数据侧同时提供 `click_label/convert_label`。
