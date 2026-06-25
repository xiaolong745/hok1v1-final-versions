# R5.6、R8 与 R8 Target Attention 对比

## R5.6.2

R5.6.2 的设计重心是稳定。它冻结了 R5.3 以来的观测和模型契约，避免在训练后期继续扩大状态空间。这个版本的核心价值在于战术语义和动作优先级：代码中显式维护清兵、推塔、必须打塔、危险撤退、技能纪律等语义，再通过 action-priority 对 logits 做小幅偏置。

它的问题也很清楚：target head 仍然是普通分类头。模型需要从一个较紧凑的特征向量中自己学出“现在该打谁”，因此对打塔、打兵、打野怪这种目标切换频繁的动作，表达能力有限。

## R8 Final

R8 final 是更高上限的版本。它把状态空间扩展到 2208 维，包含 dense scalars、discrete bins、16 个 unit slots、lane local map 和 temporal state。同时它引入多价值头，把总回报拆成 objective、lane、combat、survival、resource 等部分。

R8 的主要短板不是信息不够，而是 target head 仍然没有结构化。`model.py` 中前五个动作头和最后一个 target 头都来自普通 `label_mlp(h)`。虽然后面有 action-priority 修正，但这属于 logits 后处理，不等于网络内部显式理解每个候选目标。

## R8 Target Attention

R8 target attention 是基于 R8 final 的赛后结构实验版。它保留 R8 的 2208 维输入和样本格式，只从已有 16 个 unit slots 中抽出 9 个 target 候选：

```text
target 0: 空目标
target 1: 敌方英雄
target 2: 自己
target 3-6: 敌方小兵 1-4
target 7: 敌方防御塔
target 8: 野怪/中立目标
```

对应 R8 unit slot 索引为：

```text
(-1, 1, 0, 8, 9, 10, 11, 3, 12)
```

其中 `-1` 表示空目标的零向量。模型先用 `target_entity_mlp` 编码每个候选目标，再用 `target_query_mlp` 根据当前 LSTM 公共状态生成 query，最后通过点积得到 9 个 target logits。

这个改法的意义是缩短学习链路。原来的 R8 需要从大向量里间接学出“该打塔/打兵/打野怪”；新结构让每个候选实体直接参与 target 打分，更适合 1v1 中的推塔、清兵、击杀后转推和野怪争夺。
