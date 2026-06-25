# R8 Target Attention 设计说明

## 改动范围

本版本只改 `versions/r8_target_attention/code/agent_ppo/model/model.py`。

没有改动：

- `DimConfig.DIM_OF_FEATURE`
- `Config.SERI_VEC_SPLIT_SHAPE`
- `Config.DATA_SPLIT_SHAPE`
- `Config.LABEL_SIZE_LIST`
- reward
- action-priority
- train workflow

因此训练样本格式和平台动作接口保持不变。

## 结构变化

R8 final 的 target head：

```python
target_logits = label_mlp(h)
```

R8 target attention 的 target head：

```python
target_embed = target_entity_mlp(target_units)
target_query = target_query_mlp(h)
target_logits = matmul(target_embed, target_query)
```

其中 `target_units` 直接来自 R8 已有的 16 个 unit slots，不额外增加 observation 维度。

## 目标顺序

本版本采用的 target 顺序为：

```text
0 none
1 enemy hero
2 self
3 enemy minion 1
4 enemy minion 2
5 enemy minion 3
6 enemy minion 4
7 enemy tower
8 monster
```

这个顺序和外部项目监控中 `tower target=7` 的习惯一致，也能覆盖我们关心的打塔、清兵和野怪目标。

## 风险

这个版本不能直接加载旧 R8 checkpoint。target head 模块已经变为 `target_entity_mlp + target_query_mlp`，旧的最后一个 `hero_label5_mlp` 参数不再存在。

另外，target attention 只解决“打谁”的结构表达问题，不自动解决 reward 权重、训练时长、对手池更新和评估稳定性问题。
