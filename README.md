# HOK1v1 Final Versions Archive

这个仓库用于赛后复盘和长期查阅，保留我们认为最有价值的几个 HOK1v1 版本：

- `versions/r5_6_2/code`: R5.6.2 platform long-train 版本，从当时的云端全量脚本解包得到。
- `versions/r8_final/code`: R8 最终低固定学习率自博弈版本，来自本地最终 `code/`。
- `versions/r8_target_attention/code`: 基于 R8 final 的结构实验版，把 target head 从普通 MLP 改为 target entity attention。

原始云端粘贴脚本保存在 `patches/`，便于追溯当时的完整部署载荷。

## 版本定位

`R5.6.2` 是稳定战术语义版本。它保持较小的 896 维观测契约，重点在 `should_clear_wave`、`can_pressure_tower`、`must_hit_tower_now` 等战术语义，以及 action-priority logits bias。

`R8 final` 是最终比赛主线结构。它使用 2208 维特征、多价值头、局部兵线图、时间状态和更丰富的奖励项，优点是表达能力强，缺点是 target 选择仍然是普通 `label_mlp`。

`R8 target attention` 是赛后复盘增强版。它不改样本维度，不改 action mask，不改 reward，只把最后一个 9 类 target head 改成 query-target 匹配：

```text
public state h -> target query
R8 unit slots -> target entity embeddings
target_logits = target_embed dot target_query
```

## 验证状态

本地已做模型级前向验证：

```text
train outputs: (32,12), (32,16), (32,16), (32,16), (32,16), (32,9), (32,6)
eval logits: (2,85)
eval value: (2,6)
```

`train_test.py` 在本机仍然会因缺少 `kaiwudrl` 运行时失败，这是本地环境依赖问题，不是本仓库结构错误。

## 重要说明

`R8 target attention` 与旧 R8 checkpoint 不兼容。它适合从头训练或作为赛后结构复盘，不适合直接加载旧 `.pkl` 继续训练。
