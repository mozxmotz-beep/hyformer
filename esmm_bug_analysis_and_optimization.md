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
