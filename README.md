# hyformer
## 尖刺预防
第 1 周：梯度裁剪 + 学习率预热 → Loss 尖刺解决
第 2 周：时间桶细化 + 嵌入 Dropout → AUC +0.3%
第 3 周：特征交叉 + 跨域注意力 → AUC +0.8%
第 4 周：多任务扩展 (CTR+CVR) + ESMM 损失 → AUC +1.2%
第 5 周：动态 Top-K + 对比损失 → AUC +1.6%
第 6 周：生成式特征增强 + 多模态融合（可选）→ AUC +2.5%+


# 训练 Loss 曲线尖刺（Loss Spike）解决方案

> 适用场景：训练过程中 Loss 曲线出现突然飙升（如第 6794 步、第 15294 步附近 Loss 从 ~0.3 飙升至 ~1.8）

---

## 一、原因分析

| 原因 | 说明 |
|------|------|
| 学习率过大 | 梯度更新步子太大，导致参数震荡 |
| 异常数据批次（Bad Batch） | 某个 batch 含有噪声或异常样本 |
| 梯度爆炸 | 梯度值过大，导致参数剧烈变化 |
| 数据加载问题 | 某些 step 的数据分布异常 |

---

## 二、解决方案

### 方案一：降低学习率 ⭐（最常见、最有效）

将学习率降低 2~5 倍，减少每步参数更新幅度。

```python
# 将学习率降低 2~5 倍
optimizer = Adam(lr=1e-4)  # 原来可能是 5e-4
```

---

### 方案二：添加梯度裁剪（Gradient Clipping）⭐

限制梯度的最大范数，防止单步更新过大。

```python
# PyTorch
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# TensorFlow / Keras
optimizer = Adam(clipnorm=1.0)
```

---

### 方案三：使用学习率预热（Warmup）

前 N 步线性增大学习率，避免初期训练震荡。

```python
from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,
    num_training_steps=total_steps
)
```

---

### 方案四：检查并清洗数据

排查训练集中的异常样本（NaN、极端值、标签错误等）。

```python
# 删除空值
df = df.dropna()

# 过滤极端值
df = df[df['feature'].between(lower_bound, upper_bound)]

# 检查标签是否合法
assert df['label'].isin([0, 1]).all(), "存在非法标签！"
```

---

### 方案五：减小 Batch Size

大 batch 对异常样本更敏感，适当减小 batch size 可降低单批次影响。

```python
# 尝试从大 batch 降低至 32 或更小
batch_size = 32
```

---

### 方案六：添加损失异常跳过机制

在训练循环中检测异常 loss，若超过阈值则跳过该 batch。

```python
threshold = 2.0  # 根据正常 loss 范围设定

for batch in dataloader:
    loss = model(batch)
    
    if loss.item() > threshold:
        print(f"[WARNING] 跳过异常 batch，loss={loss.item():.4f}")
        optimizer.zero_grad()
        continue
    
    loss.backward()
    optimizer.step()
```

---

## 三、方案优先级推荐

| 优先级 | 方案 | 效果 | 实现难度 |
|--------|------|------|----------|
| ⭐⭐⭐ | 梯度裁剪 | 高 | 低（一行代码） |
| ⭐⭐⭐ | 降低学习率 | 高 | 低 |
| ⭐⭐ | LR Warmup | 中 | 中 |
| ⭐⭐ | 数据清洗 | 中 | 中 |
| ⭐ | 减小 Batch Size | 低~中 | 低 |
| ⭐ | 异常 Batch 跳过 | 辅助 | 低 |

---

## 四、推荐首选组合

> 同时启用以下两项，通常可解决 90% 的 Loss 尖刺问题：

```python
# 1. 降低学习率
optimizer = Adam(lr=1e-4)

# 2. 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

如果尖刺仍然存在，再逐步排查数据质量问题或引入 Warmup 策略。

---

## 五、参考资料

- [PyTorch Gradient Clipping 文档](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Hugging Face Warmup Scheduler](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
- Goodfellow et al., *Deep Learning*, Chapter 8: Optimization for Training Deep Models

