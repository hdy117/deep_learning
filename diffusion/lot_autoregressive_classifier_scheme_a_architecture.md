# 彩票自回归分类方案 A 软件架构说明

## 1. 文档目的

本文档说明 `diffusion/lot_autoregressive_classifier_scheme_a.py` 的软件架构、核心模块、数据流、训练与采样流程、设计取舍，以及后续可扩展方向。

它的定位不是简单的“代码备注”，而是面向后续维护、调参、对比 diffusion 方案、继续扩展评估能力时的设计说明书。

对应实现文件：

- `diffusion/lot_autoregressive_classifier_scheme_a.py`

相关参考文件：

- `diffusion/lot_ddpm.py`
- `diffusion/lot_ddpm_diffusers_ai.py`
- `diffusion/data/lot_data.csv`

---

## 2. 背景与设计目标

当前项目中已有的彩票脚本主要使用扩散/连续值回归思路：

- 将号码映射到连续空间；
- 训练时预测噪声或连续目标；
- 采样后再做 round / clip / 去重过滤。

这种方式的问题在于：

1. 彩票号码本质上是离散类别，不是连续值；
2. 红球标签在数据中天然有顺序语义：`red_ball_0 < red_ball_1 < ... < red_ball_5`；
3. 扩散方案往往要依赖后处理才能过滤非法样本；
4. 训练目标和最终合法输出规则之间存在偏差。

因此，方案 A 的核心设计目标是：

1. **离散化建模**
   - 直接把号码当作分类标签处理。

2. **顺序化生成**
   - 按红球 1 到红球 6，再到蓝球的顺序生成。

3. **规则内生化**
   - 在采样过程中直接施加“红球严格递增、不可重复”的约束。

4. **保持项目兼容性**
   - 复用现有脚本的组织方式、训练入口、采样入口和输出文件模式。

5. **便于横向比较**
   - 新脚本可与现有 `lot_ddpm*.py` 并行运行、并行评估。

---

## 3. 设计原则

本脚本遵循以下实现原则：

### 3.1 简单优先

模型结构优先选择：

- PyTorch 原生模块；
- 便于阅读与调试；
- 尽量少引入外部复杂依赖。

### 3.2 与任务结构对齐

模型结构不追求“通用生成模型”的大而全，而是尽量贴合彩票任务的特点：

- 输入是固定长度历史窗口；
- 输出是固定长度的 7 个离散 token；
- 红球有单调递增约束；
- 蓝球是单独的类别空间。

### 3.3 训练目标与采样规则一致

训练时：

- 按 CSV 中天然顺序学习红球标签；
- 按 token 序列顺序做 teacher forcing。

采样时：

- 按同样顺序生成；
- 红球位置约束与标签定义保持一致。

### 3.4 保持现有工程使用习惯

继续沿用：

- `Config` 统一管理超参数；
- `--train` / `--sample` CLI；
- 每行一个 Python list 的 txt 输出方式；
- 与原脚本接近的日志与 checkpoint 使用模式。

---

## 4. 总体架构视图

脚本从职责上可以拆成 7 个层次：

1. **基础设施层**
   - 设备选择
   - 随机种子
   - 位置编码

2. **数据层**
   - CSV 读取
   - 滑窗样本构造
   - 时序切分
   - DataLoader 组织

3. **历史条件编码层**
   - `HistoryEncoder`
   - 将过去 72 期编码为条件记忆

4. **自回归解码层**
   - `AutoregressiveLotteryModel`
   - 使用 TransformerDecoder 输出位置级分类 logits

5. **训练控制层**
   - `compute_loss()`
   - `validate()`
   - `train()`

6. **采样控制层**
   - `sample_from_logits()`
   - `sample()`
   - 合法性 mask 逻辑

7. **输出持久化层**
   - `append_list_to_file()`
   - txt 文件输出

---

## 5. 高层模块关系图

```text
                    +--------------------------------------+
                    | lot_data.csv                         |
                    | 历史开奖数据                         |
                    +-------------------+------------------+
                                        |
                                        v
                    +--------------------------------------+
                    | LotAutoregressiveDataset             |
                    | 滑窗构造: 72期历史 -> 下1期标签      |
                    +-------------------+------------------+
                                        |
                                        v
                    +--------------------------------------+
                    | chronological_split                  |
                    | 时间顺序划分 train / gap / val       |
                    +----------+----------------+----------+
                               |                |
                               v                v
                    +----------------+   +----------------+
                    | train_loader    |   | val_loader      |
                    +--------+--------+   +--------+-------+
                             |                     |
                             +----------+----------+
                                        |
                                        v
                    +--------------------------------------+
                    | AutoregressiveLotteryModel           |
                    | 历史编码 + Transformer自回归解码     |
                    +-------------------+------------------+
                                        |
                        +---------------+----------------+
                        |                                |
                        v                                v
             +----------------------+        +----------------------+
             | train / validate     |        | sample               |
             | 训练、验证、存权重   |        | 约束解码、生成号码   |
             +----------+-----------+        +----------+-----------+
                        |                                 |
                        v                                 v
             +----------------------+        +----------------------+
             | checkpoint .pth      |        | output .txt          |
             | 最优模型参数         |        | 每行一个号码列表     |
             +----------------------+        +----------------------+
```

---

## 6. 数据架构

## 6.1 原始数据格式

数据来自：

- `diffusion/data/lot_data.csv`

使用字段：

- `red_ball_0`
- `red_ball_1`
- `red_ball_2`
- `red_ball_3`
- `red_ball_4`
- `red_ball_5`
- `blue_ball_0`

约定：

- 红球已按升序存储；
- 蓝球独立；
- 每一行表示一期开奖结果。

### 6.2 数据语义

每一行实际上是一个长度为 7 的离散结构化事件：

```text
[red1, red2, red3, red4, red5, red6, blue]
```

其中：

- `red1..red6` 属于集合 `1..33`
- `blue` 属于集合 `1..16`

这不是一个连续向量，而是一个带约束的离散对象。

---

## 7. 数据集设计

## 7.1 滑窗构造

`LotAutoregressiveDataset` 使用滑窗方式构造监督样本：

- 输入 `history`: 最近 72 期，形状 `[72, 7]`
- 输出标签：第 73 期
  - `target_red`: `[6]`
  - `target_blue`: 标量

### 构造示意

```text
第 0 个样本:
  输入 = rows[0:72]
  标签 = rows[72]

第 1 个样本:
  输入 = rows[1:73]
  标签 = rows[73]

...
```

### 7.2 标签映射

为了适配 `CrossEntropyLoss`：

- 红球从 `1..33` 映射到 `0..32`
- 蓝球从 `1..16` 映射到 `0..15`

这只发生在训练标签层，不改变原始历史输入值。

### 7.3 为什么不标准化

与 diffusion 脚本不同，这里**不做连续标准化**，原因是：

1. 分类任务不需要把类别值映射到连续区间；
2. 号码的“大小差 1”不应被解释为连续几何距离；
3. embedding 更适合表达离散类别语义。

---

## 8. 训练/验证切分设计

## 8.1 顺序切分

`chronological_split()` 使用时间顺序划分：

- 前 80% 样本用于训练
- 后 20% 样本用于验证

### 8.2 gap 机制

若样本量允许，在训练集和验证集之间保留 `seq_length` 大小的 gap。

目的是减少窗口重叠：

- 否则训练最后一个窗口与验证第一个窗口会共享大量历史片段；
- 这种重叠会导致验证结果偏乐观。

### 8.3 为什么不使用 random_split

因为这是时间序列预测问题，而不是 i.i.d. 样本问题：

- 随机打散会破坏时间顺序；
- 会让未来信息间接泄漏到训练中；
- 也不符合真实使用场景。

---

## 9. 模型架构总览

模型由两部分组成：

1. `HistoryEncoder`
2. `AutoregressiveLotteryModel` 中的 TransformerDecoder

可以理解为一个典型的 encoder-decoder 结构：

```text
history (72 draws)
   -> HistoryEncoder
   -> history_memory + history_context
   -> TransformerDecoder(prefix tokens, cross-attend history_memory)
   -> red_head / blue_head
```

---

## 10. HistoryEncoder 详细设计

## 10.1 输入格式

输入：

- `history`: `[B, 72, 7]`

其中每个 draw 包含 7 个离散号码。

## 10.2 embedding 设计

为了区分不同语义来源，使用三类 embedding：

1. **red_embedding**
   - 编码红球取值 `1..33`

2. **blue_embedding**
   - 编码蓝球取值 `1..16`

3. **field_embedding**
   - 编码字段位置：
     - `red_ball_0`
     - `red_ball_1`
     - ...
     - `red_ball_5`
     - `blue_ball_0`

这样模型不仅知道“号码是什么”，也知道“号码位于哪个字段”。

## 10.3 单期聚合

一期开奖结果有 7 个号码 embedding，当前实现采用：

- 将 7 个 embedding 取均值，得到单期 draw 向量。

这样设计的原因：

- 简单稳定；
- 参数少；
- 对当前任务足够直接；
- 避免过早引入 draw 内复杂结构。

## 10.4 历史序列编码

72 个 draw 向量进一步经过：

- 正弦位置编码
- `TransformerEncoder`

输出：

- `history_memory`: `[B, 72, d_model]`
- `history_context`: `[B, d_model]`

其中：

- `history_memory` 供 decoder cross-attention 使用；
- `history_context` 是对历史的全局摘要。

---

## 11. 自回归 token 设计

## 11.1 统一词表

decoder 使用统一 token 词表：

- `0 = BOS`
- `1..33 = red_1..33`
- `34..49 = blue_1..16`

### 优点

1. 采样阶段所有步骤都能复用同一套 prefix 机制；
2. 训练时 teacher forcing 输入组织简单；
3. 目标序列可统一表示为 token 序列。

## 11.2 目标序列定义

目标开奖：

```text
[red1, red2, red3, red4, red5, red6, blue]
```

映射成 token 后就是长度为 7 的目标序列。

## 11.3 BOS 的作用

BOS 是序列起始标记：

- 第一步输入 BOS，预测第一个红球；
- 后续位置输入前序真实 token 或已采样 token。

---

## 12. TransformerDecoder 详细设计

## 12.1 输入组成

decoder 输入由三部分组成：

1. token embedding
2. 位置编码
3. 全局 history context 偏置

即：

```text
decoder_input = token_embedding(prefix)
              + positional_encoding
              + projected_history_context
```

## 12.2 causal mask

使用上三角 mask，保证 decoder 的自注意力满足自回归约束：

- 当前位置不能看到未来 token；
- 只能依赖 BOS 和已有前缀。

## 12.3 cross-attention

decoder 通过 cross-attention 读取 `history_memory`：

- 这样每个生成位置都可以根据过去 72 期历史信息进行条件预测；
- 比只用一个 pooled context 更灵活。

## 12.4 输出层

decoder 输出的每个位置隐藏向量再进入分类头：

- 红球位置走 `red_head`
- 蓝球位置走 `blue_head`

这是“共享主干 + 分头预测”的结构。

---

## 13. 训练流程设计

## 13.1 总体流程

```text
加载 Config
   -> 构建 dataset / split / dataloader
   -> 构建 model / optimizer / scheduler
   -> epoch 循环
       -> batch 前向
       -> 计算 loss
       -> backward
       -> grad clip
       -> optimizer step
   -> validate
   -> 若 val 更优则保存 checkpoint
```

## 13.2 Teacher Forcing 机制

目标 token 序列设为：

```text
y = [red1, red2, red3, red4, red5, red6, blue]
```

decoder 输入是：

```text
[BOS, red1, red2, red3, red4, red5, red6]
```

模型预测：

- 第 1 个位置 -> `red1`
- 第 2 个位置 -> `red2`
- ...
- 第 7 个位置 -> `blue`

这种方式有两个优点：

1. 训练稳定，收敛速度快；
2. 直接对应采样时的逐位生成逻辑。

## 13.3 Loss 设计

总损失由 7 个位置的 CE loss 取平均：

```text
loss = (loss_red_1 + loss_red_2 + ... + loss_red_6 + loss_blue) / 7
```

为什么不用单一 49 类统一 head：

- 红球和蓝球语义不同；
- 蓝球类别空间更小；
- 分开 head 更清晰，也更方便后续调权重。

## 13.4 梯度裁剪

加入 `clip_grad_norm_`：

- 避免训练中梯度过大；
- 对 Transformer 训练更稳妥；
- 成本低、收益明确。

## 13.5 best checkpoint 策略

以验证集 loss 为准保存：

- 当前最优时写入 checkpoint
- 路径：`diffusion/models/lot_autoregressive_classifier_scheme_a.pth`

这比仅按 epoch 间隔保存更适合当前实验场景。

---

## 14. 验证流程设计

验证阶段：

- 不做采样；
- 直接用 teacher forcing 前向；
- 计算平均分类 loss。

这样做的目的：

- 让训练与验证指标一致；
- 避免把采样随机性混入模型选择；
- checkpoint 选择标准更稳定。

---

## 15. 采样流程设计

## 15.1 总体流程

```text
加载模型 checkpoint
   -> 读取最近72期历史
   -> 编码 history
   -> prefix = [BOS]
   -> 依次生成 6 个红球
   -> 再生成 1 个蓝球
   -> 输出 [r1, r2, r3, r4, r5, r6, b]
   -> 写入 txt 文件
```

## 15.2 采样流程图

```text
latest history
     |
     v
HistoryEncoder
     |
     +--> history_memory
     +--> history_context
     |
     v
prefix = [BOS]
     |
     v
step 1: predict red_1 -> apply red constraints -> sample
     |
     v
step 2: predict red_2 -> apply red constraints -> sample
     |
     v
step 3: predict red_3 -> apply red constraints -> sample
     |
     v
step 4: predict red_4 -> apply red constraints -> sample
     |
     v
step 5: predict red_5 -> apply red constraints -> sample
     |
     v
step 6: predict red_6 -> apply red constraints -> sample
     |
     v
step 7: predict blue -> sample
     |
     v
合法号码列表输出
```

## 15.3 采样控制参数

当前提供：

- `temperature`
- `top_k`

作用：

- `temperature` 控制分布平滑程度；
- `top_k` 控制只在高概率类别中采样。

这是为了兼顾：

- 随机性
- 可控性
- 后续调参空间

---

## 16. 红球合法性约束设计

这是本方案最关键的工程点之一。

## 16.1 约束一：不可重复

已选择的红球不能再次选择。

## 16.2 约束二：严格递增

当前红球必须大于前一个红球。

例如：

- 如果上一个红球是 12
- 当前合法值只能在 `13..33`

## 16.3 约束三：为后续位置留空间

假设：

- 当前正在生成第 4 个红球
- 后面还剩 2 个红球需要生成

那么当前值不能大于 31，否则后面无法再找到两个更大的不同号码。

### 这个约束为什么必要

如果只约束“递增 + 不重复”，仍然可能在前面步骤把号码选得太大，导致后面无合法值可选。

这个“剩余空间约束”能从根本上避免生成死路。

## 16.4 蓝球为什么不参与这些约束

因为蓝球和红球不属于同一规则集合：

- 蓝球取值空间独立；
- 蓝球不需要和红球去重；
- 蓝球不需要满足递增性质。

---

## 17. 输出与文件架构

## 17.1 checkpoint

模型权重保存到：

- `diffusion/models/lot_autoregressive_classifier_scheme_a.pth`

## 17.2 采样输出

生成号码写到：

- `diffusion/lot_autoregressive_classifier_scheme_a.txt`

每行一个 Python list，例如：

```python
[1, 6, 12, 20, 25, 30, 7]
```

### 保持这种格式的原因

- 与现有 diffusion 输出格式一致；
- 方便已有评估脚本复用；
- 便于人工快速检查合法性。

---

## 18. CLI 与运行方式

脚本保留现有项目风格：

### 18.1 训练

```bash
python diffusion/lot_autoregressive_classifier_scheme_a.py --train
```

### 18.2 采样

```bash
python diffusion/lot_autoregressive_classifier_scheme_a.py --sample
```

### 18.3 运行约束

由于模型结构从旧版 GRU 改为 Transformer：

- 旧 checkpoint 与新结构不兼容；
- 需要重新训练生成新 checkpoint；
- 脚本会在采样时给出明确提示，而不是直接无提示失败。

---

## 19. 配置架构

`Config` 负责统一管理：

### 19.1 路径配置

- `data_path`
- `model_path`
- `out_file`

### 19.2 数据参数

- `cond_seq_length`
- `batch_size`
- `val_batch_size`
- `train_ratio`
- `split_gap`

### 19.3 模型参数

- `d_model`
- `nhead`
- `num_layers`
- `dropout`

### 19.4 优化参数

- `lr`
- `epochs`
- `weight_decay`
- `gradient_clip`

### 19.5 采样参数

- `sample_batch_size`
- `temperature`
- `top_k`

把这些参数集中在 `Config` 的好处是：

- 便于实验复现；
- 便于与其他脚本对比；
- 便于后续把配置迁移到 yaml/json 或实验管理框架。

---

## 20. 与 diffusion 方案的架构对比

## 20.1 表达空间对比

### diffusion 方案

- 连续值空间
- 训练预测噪声
- 最终再映射回离散号码

### 自回归分类方案

- 离散 token 空间
- 训练直接预测类别
- 输出天然落在合法类别空间

## 20.2 合法性控制对比

### diffusion 方案

- 采样后过滤非法结果
- 红球顺序与唯一性更多依赖后处理

### 自回归分类方案

- 生成过程中直接施加约束
- 合法性是解码逻辑的一部分

## 20.3 可解释性对比

### diffusion 方案

- 中间过程更偏潜变量去噪
- 解释每个输出位置的生成依据相对困难

### 自回归分类方案

- 每一步就是“根据历史和前缀预测下一个球”
- 更贴近人类对顺序生成的直觉

---

## 21. 当前实现的取舍与限制

## 21.1 当前优点

1. 任务结构对齐更好
2. 合法性控制更直接
3. 实现清晰，便于维护
4. 训练/采样逻辑对称

## 21.2 当前限制

1. **draw 内部建模较简单**
   - 单期 7 个号码目前用均值聚合；
   - 还没有显式建模单期内部更复杂关系。

2. **采样仍是局部逐步采样**
   - 没有 beam search；
   - 没有全局最优序列搜索。

3. **当前文档和代码以实验为中心**
   - 还不是一个拆分成 package 的正式工程。

4. **验证指标仍较基础**
   - 目前主要看 cross-entropy；
   - 尚未纳入更面向业务的统计指标。

---

## 22. 后续扩展建议

### 22.1 模型结构扩展

1. draw 内部增加 self-attention
2. decoder 增加 token type embedding
3. 历史编码中加入 class token 或 attention pooling
4. 增加更强的 regularization

### 22.2 采样策略扩展

1. top-p 采样
2. beam search
3. 多样性惩罚
4. 重复模式抑制

### 22.3 评估系统扩展

1. 输出分布与历史分布的差异分析
2. 红球边际分布统计
3. 蓝球边际分布统计
4. 序列多样性评估
5. 与 diffusion 输出统一对比报告

### 22.4 工程化扩展

1. 将模型、数据集、训练器拆分为独立模块
2. 增加配置文件化
3. 增加实验编号与结果归档
4. 增加自动评估脚本

---

## 23. 维护建议

后续维护时，建议优先关注以下几个稳定性点：

1. **不要破坏红球标签顺序定义**
   - 训练和采样都默认红球是升序标签。

2. **不要改回 random_split**
   - 否则验证结果会掺入时间泄漏。

3. **不要让蓝球参与红球唯一性约束**
   - 这是两个独立类别空间。

4. **改 decoder 结构时注意 checkpoint 兼容性**
   - 结构变化后需要重新训练。

5. **若改输出格式，需同步评估链路**
   - 当前 txt 输出格式是为了兼容已有脚本。

---

## 24. 训练流程图

```text
启动 --train
   |
   v
读取 Config
   |
   +--> 加载 CSV
   +--> 构造滑窗数据集
   +--> 时间顺序划分 train/val
   +--> 构造 DataLoader
   +--> 初始化 Transformer 模型
   +--> 初始化 optimizer / scheduler
   |
   v
epoch loop
   |
   v
for batch in train_loader
   |
   +--> encode history
   +--> build teacher-forcing prefix
   +--> TransformerDecoder 前向
   +--> red_head / blue_head 输出 logits
   +--> 计算 7 个位置 CE loss
   +--> backward
   +--> grad clip
   +--> optimizer.step()
   |
   v
validate
   |
   v
若 val loss 更优 -> 保存 checkpoint
   |
   v
训练结束 -> 保存 loss 曲线图
```

---

## 25. 采样流程图

```text
启动 --sample
   |
   v
读取 Config
   |
   v
加载 checkpoint
   |
   v
读取最近 72 期 history
   |
   v
HistoryEncoder 编码
   |
   +--> history_memory
   +--> history_context
   |
   v
prefix = [BOS]
   |
   v
循环 6 次生成红球
   |
   +--> decoder 输出 red logits
   +--> 应用合法性 mask
   |      - 不重复
   |      - 严格递增
   |      - 保留剩余空间
   +--> 采样 1 个红球
   +--> 追加到 prefix
   |
   v
decoder 输出 blue logits
   |
   +--> 采样 1 个蓝球
   |
   v
拼成 [6 red + 1 blue]
   |
   v
打印并写入 txt
```

---

## 26. 总结

该方案的本质是：

- 用 TransformerEncoder 编码过去 72 期历史；
- 用 TransformerDecoder 按固定顺序自回归生成下一期号码；
- 用分类替代连续回归；
- 用解码阶段约束替代采样后过滤。

因此，这个方案相比 diffusion 思路，更贴近该任务本身的结构：

- **离散**：号码是类别，不是连续值；
- **顺序**：红球位置有天然顺序；
- **约束**：合法性规则可以内嵌到生成过程中；
- **工程可维护性**：结构清晰，便于继续调参与扩展。

如果后续继续演进，这个脚本很适合作为“彩票离散序列生成”的主实验基线。