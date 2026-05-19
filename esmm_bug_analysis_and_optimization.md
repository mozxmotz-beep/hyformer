
# esmm_bug 分支：方案检查 + 代码结构分析 + 可执行稳定落地方案

> 目标：把当前“单 logit 训练”的 ESMM 退化实现，升级为可稳定运行、可监控、可灰度回滚的多任务 CTR/CVR/CTCVR 训练框架。


## 0. 执行摘要（结论先行）

当前分支存在的瓶颈并不是“模型层数不够”，而是**训练目标与数据生成机制不一致**：

1. 仅训练 `ctcvr_logit`，没有独立监督 CTR/CVR，ESMM 实质退化。  
2. CTR/CVR 共用最后同一 representation，负迁移明显。  
3. CVR 是 clicked-only 观测任务，但结构没有 click-aware 路径。  
4. time bucket 仅做加法偏置，缺少行为强度随时间衰减机制。

因此建议按三阶段落地：
- **Phase 1（必须先做）**：纠正 loss 与输出协议（恢复 ESMM 训练闭环）。
- **Phase 2**：用 MMoE/CGC 做任务表征解耦。
- **Phase 3**：引入 click-aware query 与时序乘性衰减提升上限。


## 1. 当前项目分支结构检查（面向可改造性）

## 1.1 关键模块职责

- `model.py`
  - 包含主模型 `PCVRHyFormer`、`MultiSeqQueryGenerator`、多序列 block、CTR/CVR heads。
  - 当前 `forward()` 最终只返回 `ctcvr_logit`，并在内部使用 `ctr_head` 与 `cvr_head` 相乘耦合。
- `trainer.py`
  - 训练主循环与 `_train_step()`。
  - 当前仅对 `self.model(model_input)` 的单 logit 做 BCE/Focal。
- `dataset.py`
  - 负责构造 batch 字段（含序列、长度、time_bucket、label）。
  - 需确认是否已有 click label；若无需补齐。
- `train.py`
  - 参数入口与 trainer/model 初始化。
  - 需要新增多任务 loss 权重与开关参数，支持灰度。
- `utils.py`
  - 目前有 focal loss 与早停等工具。
  - 可扩展多任务指标汇总工具（可选）。

## 1.2 当前“可直接改造”的优势

- 已有 `ctr_head/cvr_head`，不是从零开始。
- 已有 `predict_esmm()`，便于扩展返回结构。
- time bucket 已进模型，增加 decay 系数改动路径清晰。

## 1.3 当前“必须补齐”的缺口

- 缺少“CTR 标签 vs 转化标签”的清晰训练协议。
- 评估端只有单任务指标，无法诊断分任务退化。
- 缺少 feature flag（无法安全灰度）。



## 2. 方案检查（逐项对齐你给出的改进点）

## 2.1 改进点A：给 CTR/CVR 独立 representation

**检查结论：必要且高优先级。**  
建议采用“共享底座 + 任务分流”的最小可行实现：
- Phase 1.5：先加 `ctr_tower` / `cvr_tower`（轻量 2-layer MLP），不立刻上 MMoE。
- Phase 2：替换为 MMoE/CGC。

这样做的价值：
- 风险低：先验证“分流是否带来增益”；
- 可归因：收益来源清楚（loss 修正 vs 表征解耦）。

## 2.2 改进点B：CTR/CVR 独立 loss（最关键）

**检查结论：必须第一优先落地。**  
推荐统一接口：模型返回 dict：
- `ctr_logit`
- `cvr_logit`
- `ctcvr_logit`

训练损失：
- `L_ctr = BCE(ctr_logit, y_click)`
- `L_cvr = BCE(cvr_logit[clicked], y_ctcvr[clicked])`
- `L_ctcvr = BCE(ctcvr_logit, y_ctcvr)`
- `L_total = w_ctr*L_ctr + w_cvr*L_cvr + w_ctcvr*L_ctcvr`

默认建议：`w_ctr=1.0, w_cvr=1.0, w_ctcvr=0.5`。

## 2.3 改进点C：CVR 序列偏差（click-aware query）

**检查结论：合理，但应在 Phase 1 稳定后再上。**

建议最小落地路径：
1. 先在 `MultiSeqQueryGenerator` 中复制一套 CVR 专用 query 参数（与 CTR 分离）。
2. 加入 click signal（若无显式点击序列，可先用 click label 生成 sample-level gate）。
3. 再逐步升级为 token-level 点击路径。

## 2.4 改进点D：time bucket 从加法改为乘性衰减

**检查结论：建议实施，且改动可控。**

优先方案（稳定优先）：
- 学习 `decay_table[bucket] -> scalar in (0,1]`（sigmoid 映射）。
- token 融合：`x_t = emb_t * decay_t + time_emb_t`。

理由：
- 比 `exp(-alpha*delta)` 更稳（不依赖外部 delta 标定）。
- 不改 attention API，工程风险更低。


## 3. 三阶段可执行落地方案（含文件级改造清单）

## Phase 1：修正目标函数与训练协议（必须）

### P1-1 数据协议梳理（dataset/trainer）
- **目标**：确保 batch 中有 `click_label` 和 `label`（`label` 作为 ctcvr）。
- 文件：`dataset.py`, `trainer.py`
- 输出：
  - `device_batch['click_label']`（若暂无，先从现有字段映射/补充）
  - `device_batch['label']`（ctcvr）

### P1-2 模型输出协议改造（model）
- 文件：`model.py`
- 新增：
  - `forward_esmm(inputs) -> dict`
  - `forward(inputs)` 保持兼容（默认返回 `ctcvr_logit`，由开关控制）
- 要求：保证线上接口不被一次性破坏。

### P1-3 多任务 loss 实现（trainer）
- 文件：`trainer.py`
- 改造 `_train_step()`：
  - 支持 `--esmm_multitask_loss` 开关
  - 计算 `L_ctr/L_cvr/L_ctcvr`
  - batch 无 clicked 样本时，`L_cvr=0` 且记录计数

### P1-4 验证与日志（trainer/train）
- 文件：`trainer.py`, `train.py`
- 新增指标：CTR/CVR/CTCVR 的 AUC + LogLoss。
- TensorBoard 前缀建议：
  - `AUC/ctr`, `AUC/cvr`, `AUC/ctcvr`
  - `Loss/ctr`, `Loss/cvr`, `Loss/ctcvr`, `Loss/total`

### P1 验收门槛
- 训练不报错，loss 无 NaN。
- CTR/CVR/CTCVR 三条曲线均可观测。
- 相比基线，CTCVR AUC 不下降或小幅提升，且 CVR AUC 明显更稳定。

---

## Phase 2：MMoE/CGC 任务解耦

### P2-1 引入 MMoE 主干
- 文件：`model.py`
- 在 `output_proj` 后增加：
  - `experts: List[MLP]`
  - `gate_ctr`, `gate_cvr`
  - `h_ctr`, `h_cvr`

### P2-2 升级为 CGC（可选）
- shared experts + task experts（ctr/cvr）。
- 两任务 gate 可访问 shared + 私有 expert。

### P2-3 训练稳定性措施
- gate entropy 正则（可选）
- expert dropout
- 梯度裁剪延续

### P2 验收门槛
- 与 Phase1 相比：
  - CTR AUC 与 CVR AUC 至少一项提升；
  - CTCVR AUC 不退化；
  - 训练波动不显著增大。

---

## Phase 3：CVR 偏差建模 + 时序衰减

### P3-1 Click-aware query path
- 文件：`model.py`（`MultiSeqQueryGenerator`）
- 增加：
  - `ctr_query_path`
  - `cvr_query_path`
- CVR 路径引入点击感知 gate（sample-level 起步，后续 token-level）。

### P3-2 时间乘性衰减
- 文件：`model.py`
- 在 `_embed_seq_domain` 中：
  - `decay = sigmoid(decay_emb(time_bucket_ids))`
  - `token_emb = token_emb * decay + time_embedding(time_bucket_ids)`

### P3 验收门槛
- 长窗口样本分桶评估提升（长周期用户/低活跃用户）。
- 线上延迟变化可控（<5% 为经验阈值）。

---

## 4. 稳定运行保障（必须执行）

## 4.1 配置与灰度开关
在 `train.py` 增加：
- `--esmm_multitask_loss`（默认 false，灰度开启）
- `--w_ctr --w_cvr --w_ctcvr`
- `--use_mmoe`（默认 false）
- `--use_click_aware_query`（默认 false）
- `--use_time_decay`（默认 false）

## 4.2 回滚策略
- 任一阶段异常可通过开关一键回落到 baseline。
- 代码层面保留旧 forward 输出兼容路径，防止推理侧联动崩溃。

## 4.3 数值稳定策略
- `ctcvr_prob = clamp(sigmoid(ctr)*sigmoid(cvr), 1e-6, 1-1e-6)` 后再 `logit`。
- clicked 样本为空时跳过 `L_cvr`，并记录 `zero_click_batches`。
- 保持梯度裁剪 `clip_grad_norm_`。

---

## 5. 里程碑与工期建议（可执行排期）

- **M1（1~2天）**：Phase1 数据协议+多任务 loss+指标打通。  
- **M2（2~3天）**：Phase2 MMoE 接入+稳定性调参。  
- **M3（2~3天）**：Phase3 click-aware + time-decay + 分桶评估。  
- **M4（1天）**：回归、压测、上线配置固化。

总计建议：**6~9个工作日** 完成可上线版本（不含大规模离线特征改造）。

---

## 6. 风险清单与应对

1. **点击样本稀疏导致 CVR 震荡**  
   - 应对：增大 batch、clicked 子集最小样本保护、epoch 级平滑监控。

2. **多任务权重不平衡**  
   - 应对：先固定 `1:1:0.5`，再小网格搜索；禁止一次性大范围搜索。

3. **MMoE 过拟合/门控塌缩**  
   - 应对：expert dropout + gate entropy regularization（可选）。

4. **改动跨度大难归因**  
   - 应对：严格分阶段发布与消融，不跨阶段叠加改动。

---

## 7. 最终交付标准（Definition of Done）

满足以下条件方可认为方案“严谨、完整、可执行、稳定”：

- [ ] 训练协议：CTR/CVR/CTCVR 三任务均有独立监督路径。  
- [ ] 指标协议：三任务 AUC/LogLoss 均可观测且落盘。  
- [ ] 配置协议：关键能力全部可开关、可灰度、可回滚。  
- [ ] 稳定性：无 NaN、无大面积梯度爆炸、吞吐可接受。  
- [ ] 效果：CTCVR 主指标不退化，CTR/CVR 至少一项显著改善。  
- [ ] 可维护性：核心改造点具备注释与实验记录。

---

## 8. 建议执行顺序（强约束）

**先 Phase1，再 Phase2，再 Phase3。**  
如果 Phase1 未通过验收，禁止推进后续结构升级。因为当前最大瓶颈是监督目标错误，不先修正会导致后续改造收益被掩盖。
=======
# ESMM_BUG 分支问题分析与优化方案

## 1) 当前实现的核心缺陷（按影响优先级）

### A. Loss 设计错误：只监督 CTCVR，未独立监督 CTR/CVR（最关键）
- 现状：`forward()` 返回单一 `ctcvr_logit`；训练时 `trainer` 仅对这个输出做 BCE/Focal。  
- 证据：`model.forward` 中最终只返回 `ctcvr_logit`；`trainer._train_step` 直接 `logits=self.model(...)` 再计算单一 loss。  
- 影响：
  - CTR 任务的梯度被“乘法耦合”稀释；
  - CVR 任务监督高度稀疏，且被 CTR 噪声干扰；
  - ESMM 的“显式分解学习（pCTR 与 pCVR）”目标没有真正实现。

### B. 表征共享过强：CTR/CVR 共用同一 `output` embedding
- 现状：`ctr_head` 与 `cvr_head` 都接同一个 `output_proj` 结果。  
- 影响：
  - 表征空间无法 task-specific 解耦；
  - CTR（曝光级）与 CVR（点击后）语义冲突，导致负迁移；
  - 任务头虽然分开，但 backbone 最后一层没有任务路由能力。

### C. CVR 样本选择偏差未建模（clicked-only 观察机制）
- 现状：query/encoder 对所有曝光样本统一处理，CVR tower 没有 click-aware path。  
- 影响：
  - CVR 侧在大量未点击样本上学习到“伪相关序列模式”；
  - 难以从“点击后的行为演化”中抽取有效信号。

### D. 时间特征融合过于线性：time_bucket 仅做加法注入
- 现状：`token_emb = token_emb + time_embedding(bucket)`。  
- 影响：
  - 只能表达“离散时间偏置”，不能表达“随时间衰减的行为强弱”；
  - 对远期行为抑制不足，影响时序建模精度。

---

## 2) 根因拆解（为何 AUC 不涨）

1. **优化目标错位**：
   你现在优化的是单目标 `P(conv|impression)`，不是同时优化 `P(click|imp)` 与 `P(conv|click,imp)`。这会让 ESMM 退化为“带结构先验的单头模型”。

2. **梯度路径劣化**：
   `pCTCVR = pCTR * pCVR` 下，若早期 `pCTR` 偏低，CVR 分支有效梯度被削弱；反之亦然。没有独立损失时，该问题更严重。

3. **多任务负迁移**：
   CTR 与 CVR 共享末端表征但标签机制不同（曝光监督 vs 点击后监督），共享 embedding 容易学到折中解。

4. **数据机制与模型结构不一致**：
   CVR 标签天然是“点击条件下观察到”，但结构上没体现该条件，导致 representation bias。

---

## 3) 优化总方案（建议分三阶段落地）

## Phase 1：修正 ESMM 训练目标（必须先做）

### 3.1 输出改造
- 模型前向返回：
  - `ctr_logit`
  - `cvr_logit`
  - `ctcvr_logit = logit(sigmoid(ctr_logit)*sigmoid(cvr_logit))`
- 保留 `ctcvr_logit` 供线上推断排序，但训练不再只依赖它。

### 3.2 多目标损失
设：
- `y_click ∈ {0,1}`（CTR 标签）
- `y_conv ∈ {0,1}`（曝光口径转化标签，即 CTCVR 标签）
- `mask_click = y_click`

建议损失：
- `L_ctr = BCEWithLogits(ctr_logit, y_click)`
- `L_cvr = BCEWithLogits(cvr_logit[mask_click==1], y_conv[mask_click==1])`  （仅在点击样本上）
- `L_ctcvr = BCEWithLogits(ctcvr_logit, y_conv)`
- `L_total = w1*L_ctr + w2*L_cvr + w3*L_ctcvr`

默认权重建议：`w1=1.0, w2=1.0, w3=0.5`（先稳健，再网格调参）。

> 若 batch 内无点击样本：`L_cvr=0`（或跳过该项），避免 NaN。

### 3.3 评估指标补齐
验证阶段同时监控：
- CTR AUC / LogLoss
- CVR AUC / LogLoss（点击子集）
- CTCVR AUC / LogLoss（全量）

避免只看 CTCVR AUC 掩盖分任务退化。

---

## Phase 2：任务表征解耦（提升上限）

### 3.4 共享底座替换为 MMoE / CGC
用 Expert+Gate 替换单 `output_proj`：
- `K` 个 experts（MLP）从共享语义中提取不同子空间；
- CTR gate 学习 `g_ctr(x)`，CVR gate 学习 `g_cvr(x)`；
- 任务输出分别为 `h_ctr = Σ g_ctr_k * expert_k(x)`，`h_cvr = Σ g_cvr_k * expert_k(x)`。

CGC 版本可加入 task-specific expert：
- shared experts + ctr experts + cvr experts
- ctr/cvr gate 可分别选择 shared 与各自私有 experts。

推荐起步超参：
- experts=4~8
- shared:task-specific = 1:1
- gate 温度可从 1.0 起。

### 3.5 Head 层次化
`ctr_head(h_ctr)`、`cvr_head(h_cvr)` 完全分离；
可增加轻量 residual tower（2-layer MLP + LN）提升任务表达能力。

---

## Phase 3：CVR 偏差建模 + 时序衰减（精修）

### 3.6 Click-aware Query Generator（针对 CVR）
在 `MultiSeqQueryGenerator` 增加双路查询：
- CTR path：以曝光序列为主（全量行为）
- CVR path：以点击序列/点击mask增强路径为主

实现要点：
1. 为 CVR query 额外输入 click signal（如点击事件 token、点击计数、最近点击间隔）；
2. CVR 查询 token 与 CTR 查询 token 不共享参数；
3. 可在 cross-attn 前后加入 click-conditioned gate。

### 3.7 时间衰减从“加法偏置”升级到“乘性权重”
当前：`x_t = emb_t + time_emb_t`
建议：
- `decay_t = exp(-alpha * delta_t)` 或可学习 bucket->scalar
- `x_t = emb_t * decay_t + time_emb_t`
或 attention 级别加权：
- `attn_score += log(decay_t)`（等价对远期 token 先验抑制）

建议先做 embedding-level 乘性衰减（改动小、稳定性高）。

---

## 4) 工程落地清单（按实施顺序）

1. **数据层**：确认 batch 提供 `click_label` 与 `conversion_label`（若仅有单 label，先补齐映射）。
2. **模型层**：forward/predict_esmm 改为返回三 logit；保留兼容接口。
3. **训练层**：重写 `_train_step` 多目标 loss + mask_click；加权可配置。
4. **评估层**：新增三套指标与日志面板。
5. **结构层**：接入 MMoE/CGC；逐步替换 `output_proj`。
6. **序列层**：加入 click-aware query path。
7. **时序层**：time decay 乘性融合。
8. **实验层**：按 Phase 逐步 A/B，避免一次性改动不可归因。

---

## 5) 实验与验收建议

### 5.1 消融实验矩阵
- Baseline（当前）
- + 独立三损失（Phase1）
- + MMoE/CGC（Phase2）
- + click-aware query（Phase3-1）
- + time decay（Phase3-2）

### 5.2 关键验收指标
- 主指标：CTCVR AUC、CTCVR LogLoss
- 诊断指标：CTR AUC、CVR AUC（clicked）
- 稳定性：loss 曲线方差、梯度范数、bad case 比例

### 5.3 预期收益（经验区间）
- Phase1 通常是最大增益来源（修目标）；
- Phase2 决定上限与鲁棒性；
- Phase3 对长链路行为与冷启动用户更敏感。

---

## 6) 对当前代码的定点问题映射

- `model.py`：`forward()` 当前只返回 `ctcvr_logit`，需要返回多任务输出。  
- `model.py`：`ctr_head/cvr_head` 共用同一 `output` 表征，需要在 head 前增加任务分流层（MMoE/CGC）。  
- `trainer.py`：`_train_step` 仅单一 BCE/Focal，需要改为 `L_ctr + L_cvr + L_ctcvr` 组合。  
- `model.py`：time_bucket 仅加法注入，需要加入乘性衰减逻辑。  
- `model.py`（`MultiSeqQueryGenerator`）：需新增 CVR click-aware query path。

---

## 7) 风险与规避

- 风险1：多损失权重不当导致某任务主导。  
  - 规避：先固定 `w1:w2:w3=1:1:0.5`，再做小步网格；可配动态 reweight。  
- 风险2：CVR 点击样本过少，batch 抖动大。  
  - 规避：clicked-only loss 用 moving average 或增大 batch；必要时分桶采样。  
- 风险3：MMoE 参数增加导致过拟合。  
  - 规避：expert dropout、L2、早停、限制 experts 数。

---

## 8) 结论

你指出的问题判断是准确的：**AUC 不涨的主因是 ESMM 被错误训练成“单 logit 单监督”**。建议先完成 Phase1（独立监督）再推进 MMoE/CGC 与 click-aware/time-decay，这样能在可解释和可归因的路径下稳定提升效果。

