# 一个节点、多种模式：H3 AdaLN 统一节点设计

目标：把 Record / Save / Load / Apply / Merge 五个节点合成一个 `H3 AdaLN`，
表统一放在 `models/h3_exact_adaln/`，用时下拉选表，连线保持「LoRA 之后 → 采样器之前」。

---

## 1. 先定不变式

```
modulation = adaln_proj( time_embedder(t) )
```

右边只依赖两样东西：**AdaLN 权重**（含打在它上面的 LoRA）和 **t**。
不含 pass、latent、分辨率、步数、sampler、conditioning、seed。

由此推出三条，整个设计都建立在它们之上：

1. 表就是一张 `t -> 调制向量` 的查找表，**按 t 索引，不按调用顺序**。
2. 同一模型在同一工作流里跑几段采样都无所谓：t 并集变大，表变多几行，仍然正确。
3. 换模型就换了 `adaln_proj`，表失效；换 AdaLN LoRA 同理。

**所以「分表」的维度是模型（+ AdaLN LoRA），不是 pass。**

## 2. 二次采样 / 高模低模 / 连续流程

### 2.1 同模型两段（二采、refine、hires fix）

一张表。record 时两段的 t 都会被记下来（`rows` 按 t 去重），apply 时两段各自命中。
**不要**为此做两张表：任何一张都只覆盖一半 t，另一段必然 t-set mismatch。

前提是同一个节点实例挂在同一个 MODEL 链上，且两段的 sigma 网格都经过它。

### 2.2 不同模型（refiner、SelfLift 高/低模、SequenceForge 的多导演台）

两个节点、两张表。判据不是「第几段」，而是「哪个 MODEL 对象」：

- 节点 A 挂在 model_high 上 → 表里存 model_high 的调制；
- 节点 B 挂在 model_low 上 → 另一张表；
- 两个节点各自只捕获取样时真正经过它的那次 forward。

因为 `adaln_proj` 是模型自己的模块，hook 不会串台。**这一点已经由现有实现保证**：
hook 周期随 MODEL 对象走，`_clear_registered` 也只清自己那个 patcher。

### 2.3 连续过程（SelfLift 这类多尺度反复调用）

能读到，条件三条：

| 条件 | 说明 | 不满足时 |
|---|---|---|
| 走 Python `forward` | hooks 挂在 `adaln_proj.linear` 上 | 表空，`rows=0`，节点应报错而不是静默存空表 |
| t 可复现 | record 与 apply 的 t 逐位相同，或差 < `1e-6 * |key|` | t-set mismatch，报错并列出缺失值 |
| 同一模型 | 高/低模各自一张表 | **静默错值**：匹配到行但语义不对 |

第三条是唯一会「跑通但结果不对」的，所以表元数据里必须写**模型指纹**，
apply 时比对，不符硬报错。

模型指纹的取法（低成本、稳定）：

```
sha1( use_adaln_curves, len(blocks), adaln_in_features,
      adaln_out_features, final_adaln_out, 前 3 个 block 的 adaln 权重头 4KB )
```

不读全量权重，几十毫秒。同一份 checkpoint 多次加载指纹一致；
同架构不同权重（比如 fl2va full vs ref2va full）必须不同——
因此要把权重字节算进去，不能只算形状。

### 2.4 唯一真正的冲突：同 t、不同值

同一模型、同一个 t 被记到两次但调制不同。触发场景：

- 运行中改了 AdaLN LoRA（比如两个 LoraLoader 串在不同分支）；
- `time_embedder_from` 与 LoRA 混用；
- 模型被中途替换。

recorder 已经在统计 `repeat_conflicts` 与 `max_repeat_delta`。
新节点把它升级为**保存前硬校验**：`max_repeat_delta > 1e-6` 就拒绝保存并报出冲突的 t。

## 3. 节点契约

```
输入
  model        MODEL
  mode         [apply | record | hybrid | merge | off]
  table        COMBO(models/h3_exact_adaln)      apply / hybrid / merge 用
  table_b      COMBO(...)                        hybrid 用
  split        INT (默认 25)                      hybrid 用
  final_from   [a | b]                           hybrid 用
  strictness   [strict | fallback]               与现有 Apply 语义一致
  table_text   STRING                            非空则覆盖下拉（应对下拉不刷新的版本）
  table_source [auto | dropdown | last]          选表策略，auto 为默认（见 §4.1）
  record_name  STRING                            留空则自动命名
  model_tag    STRING                            record 时写进名字
  lora_tag     STRING                            record 时写进名字
  pass_tag     STRING                            可选；一个工作流录两个模型时用来区分（见 §4.2）
  table_in     H3_ADALN_TABLE (可选)             同一次运行内做 A/B 时用
  after        * (可选)                          同上

输出
  model        MODEL
  table        H3_ADALN_TABLE (可选，透传)
  report       STRING
```

- `apply`：`model -> node -> sampler`，其余不用连。
- `record`：同样 `model -> node -> sampler`，表在**采样结束后自动落盘**，
  不需要 Save 节点、也不需要 `after`。
- `hybrid`：选两张表 + split，合并结果既落盘也直接注入当前 model。
- `merge`：只落盘，不改 model。
- `off`：透传，用于对照实验（会清掉残留 hook）。

## 4. 自动命名

```
{model}__{lora}__{scheduler}{steps}__sh{sv}-{sa}__{fp6}.safetensors
```

例：

```
b25-49full-int8__dmad8x1.0__simple8__sh12-2__001c09.safetensors
fl2va-full-int8__none__euler20__sh12-2__e3a71b.safetensors
```

- 每段只保留 `[A-Za-z0-9._-]`，超长截断到 32 字符；
  `model_tag` / `lora_tag` 留空时用 `model` / `none`。
- `fp6` = `lora_fingerprint` 前 6 位。它已经把 LoRA 文件名（间接）、
  强度、patch 数量都揉进去了，所以「同 LoRA 不同强度」不会撞名。
- `steps` 与 `scheduler` 来自实际调度；`sh` 是 `shift_video-shift_audio`。
- 文件名里**不放 t 的行数**：行数进元数据（`t_rows`）。这样同一配置重录是覆盖，
  不会堆出一堆 `t7 / t15 / t23`。

元数据里写全：`model_tag`、`model_fingerprint`、`lora_tag`、`lora_fingerprint`、
`scheduler`、`steps`、`shift_v`、`shift_a`、`t_rows`、`t_set_hash`、
`comfyui_version`、`created`、`record_mode`。

下拉栏里的显示用 `folder_paths` 的 COMBO；为兼容「COMBO 只在启动时构建」的版本，
额外保留 `table_text` 文本框，非空即覆盖。

## 4.1 记录→推理的无缝衔接（table_source）

麻烦点：刚录完就想把同一个节点切成 `apply` 用，但下拉栏可能还没刷新（ComfyUI 的
COMBO 有的版本只在启动时构建），手打名字又违背「简单」的初衷。

做法：**录完立刻把名字写进一个指针文件**，apply 时按策略解析。

- 保存成功后写 `models/h3_exact_adaln/_last_table.json`：
  `{"name": ..., "path": ..., "created": ..., "t_rows": ..., "tags": {...}}`，
  原子替换写（`os.replace`），不会读到半个文件。
- `table_source` 三档：
  - `auto`（默认）：下拉项能解析就用它；解析不了就退回 `_last_table.json`；
    两者都不可用才报错，并把候选列出来。
  - `dropdown`：只认下拉项，旧行为，适合固定引用某张表。
  - `last`：只认最近一次录制，适合「边调边用」。
- `table_text` 始终最高优先级，用来兜底/指定绝对路径。

实测（本地单测）：下拉项指向已删除的文件时，`auto` 会落到最近录制并说明原因；
`dropdown` 则直接报错。这样「录完切模式就能跑」不需要任何额外操作。

## 4.2 一个工作流里两个模型（二采 / 高低模）

节点挂在哪个 MODEL 上，就只会看到流过那个 MODEL 的 forward——hook 是挂在
`adaln_proj` 模块上的，天然不串台。所以二采换模型时：

- 两个节点，各自挂各自的模型接口；
- 各自填一个 `pass_tag`（如 `p1` / `p2`、`high` / `low`），它会进文件名和元数据，
  两张表不会撞名；
- apply 侧用 `model_fingerprint` 兜底：拿错表会直接报错，而不是悄悄出图。

这个判定**不做自动推断**，全靠节点上的 `pass_tag`。同一模型的分段采样不需要
tag：一张表覆盖两段 t 并集即可（§2.1）。

## 5. 保存时机：为什么不需要 `after`

现在要 `after` 是因为「节点执行时表还是空的」。统一节点换一个思路：

- 在 `record` 模式里注册 `CallbacksMP.ON_CLEANUP`，prompt 结束时统一落盘；
- 落盘前做三件事：`rows > 0`、`repeat_conflicts` 为 0、名字确定；
- 每个 prompt 只存一次（`_h3_saved` 标志），避免模型因显存被换出时重复写；
- 存盘成功后写 `_last_table.json`，供 §4.1 的 `auto` 使用。

失败时**不要**静默：在 report 里写明「未保存：rows=0 / 存在冲突」。
同一 prompt 内要 A/B（一边录一边用），再连 `table_in`——那是唯一需要连线的场景。

## 6. 表体积

一行 = 51 投影 × 96768 × 4B ≈ 19.7 MiB。

| t 行数 | 表大小 |
|---|---|
| 8 | ~158 MiB |
| 15 | ~296 MiB |
| 40（两段 20 步） | ~790 MiB |
| 100（SelfLift 多尺度） | ~2.0 GiB |

连续流程会产生大量 t。建议：record 时按 `round(t, 7)` 去重（已有），
并在 report 里提示行数与体积；超过 200 行给警告。

## 7. 与现有节点的关系

- 旧的 `H3 AdaLN Record / Save / Load / Apply / Merge` 全部保留，不删。
  已经发布的工作流（05/06/07/08）继续可用。
- 新节点内部直接复用 `record_engine` / `apply_engine` / `table_io`，
  不复制逻辑。
- 迁移路径：文档给出「旧五节点 → 新节点」的对照表，老用户不必改。

## 8. 实例开机后要验证的清单

1. `ON_CLEANUP` 是否真的在采样结束后触发，且只触发一次；
2. `folder_paths.add_model_folder_path("h3_exact_adaln", ...)` 后，
   下拉能否在不重启的情况下看到新存的表（不能就靠 `table_text` 兜底）；
3. 二采（两个 KSampler 串联、同一 model）→ 一张表覆盖两段 t，逐位 PASS；
4. 二采但第二段换模型 → 两个节点两张表，各自 PASS；
5. SelfLift 的图：hooks 是否触发、t 是否可复现、冲突计数是否为 0；
6. SequenceForge 的导演台图：同一模型被多处调用时，表是否仍逐位一致；
7. 模型指纹：fl2va full 与 ref2va full 必须不同指纹。

## 8.1 本轮新增、需要一起验的两条

- **记录→推理无缝衔接**：录完后不改任何参数、只把 `mode` 切到 `apply`，看是否
  自动选中刚录的那张表（`table_source=auto`，报告里会写
  `last recorded (dropdown entry not on disk yet)`）。同时确认下拉栏是否已经
  自己刷新了——如果刷新了，说明 COMBO 缓存不是问题，`table_text` 可以退化成
  纯兜底。
- **两个模型各录各的**：二采换模型时两个节点都给 `pass_tag`，确认生成的两个文件
  名不同、且 apply 侧互相拿错表会报 `model fp` 不匹配而不是出图。

## 9. 已知不解决

- 运行中改 AdaLN LoRA：仍要求重录（检测到冲突就拒绝保存）。
- C++/fused 引擎绕过 `module.forward`：hooks 挂不上，record 直接报错。
- 同一模型、同一 t、期望不同调制：物理上无解，只能报冲突。
