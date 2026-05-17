# hyformer
## 尖刺预防
第 1 周：梯度裁剪 + 学习率预热 → Loss 尖刺解决
第 2 周：时间桶细化 + 嵌入 Dropout → AUC +0.3%
第 3 周：特征交叉 + 跨域注意力 → AUC +0.8%
第 4 周：多任务扩展 (CTR+CVR) + ESMM 损失 → AUC +1.2%
第 5 周：动态 Top-K + 对比损失 → AUC +1.6%
第 6 周：生成式特征增强 + 多模态融合（可选）→ AUC +2.5%+


## 去尖刺训练策略（推荐默认）

已内置如下组合策略用于降低训练 loss / AUC 突发尖刺：

1. **下调 sparse_lr**：默认从 `0.05` 调整为 `0.01`。
2. **Warmup + Cosine**：通过 `--lr_schedule warmup_cosine`、`--warmup_ratio`、`--min_lr_ratio` 控制。
3. **可配置梯度裁剪**：`--grad_clip_norm`（`<=0` 表示关闭）。
4. **EMA 监控**：`--ema_decay` + `--ema_eval`，用于稳定评估曲线与监控尖刺风险。

推荐起始参数：

```bash
python train.py \
  --sparse_lr 0.01 \
  --lr_schedule warmup_cosine \
  --warmup_ratio 0.05 \
  --min_lr_ratio 0.1 \
  --grad_clip_norm 1.0 \
  --ema_decay 0.999 \
  --ema_eval
```
