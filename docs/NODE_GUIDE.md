# H3 AdaLN 使用教程（v2：两个节点，不用选表）

> 适用版本：v2 精简版。插件只注册三个节点：`H3 AdaLN`、
> `H3 AdaLN Hybrid`、`H3 Latent Compare`；旧的八个单功能节点、离线 Bake、
> 以及 01~08 工作流已全部删除，旧工作流不再可加载。
>
> 数值结论没有变：新节点直接复用 `record_engine` / `apply_engine` /
> `table_io`，和云实例逐位 PASS 的是同一套代码。本文只讲「怎么用」和「哪里会错」。

---

## 0. 一句话

```
全量教师 + mode=record  →  跑一次          →  表自动落盘
剪枝基座 + mode=apply   →  table=(auto)   →  自动找到对应表，逐位复刻
```

不用填标签、不用 Save/Load 节点、不用选表（第一次见 §3.3）、
也不用 `after` 连线（除非要在同一个 prompt 里既录又用，见 §4.4）。

---

## 1. 为什么可以这么简单

```
modulation = adaln_proj( time_embedder(t) )
```

**每个 t 对应的数值**只由 AdaLN 权重（含打在上面的 LoRA）和 t 决定，
不含 pass、latent、seed、conditioning。

但要分清两件事，很多误解都出在这里：

- **内容**（某个 t 对应什么向量）：只由权重 + LoRA + 精度决定；
- **覆盖**（表里存了哪些 t）：由 sigma 网格（调度器 + 步数 + 自定义 sigmas）、
  shift、以及这次 forward 里**存在哪些段**（有没有关键帧 / 参考图 / 参考音频）决定。

| 变量 | 影响每行的数值 | 影响表里有哪些 t |
|---|---|---|
| 教师权重 / AdaLN LoRA / 精度 | 是 | — |
| 步数、调度器、自定义 sigmas | 否 | **是**，整组都换（t = 1 − sigma） |
| video / audio shift | 否 | **是**，但**只换音频那 N 行**，视频行全部复用 |
| 有/无关键帧、参考图、参考音频（**第一次**） | 否 | 是，只多一行：`0.999`（图/帧）或 `1.0`（参考音频） |
| 换图、加图、减图、首帧还是尾帧、几张参考 | 否 | **否** |
| **分辨率、帧数、seed、采样器实现** | 否 | **否** |

所以「换分辨率要重录」是误解：帧数和分辨率只改变每段的 token 数，不改变
`t = 1 − sigma` 这个映射，也不改变段的种类。

两条 t 的来源不一样，这决定了改参数要补多少行：

```
t_v = 1 − sigma                     ← 只跟 sigma 网格有关，与 shift 无关
t_a = 1 − shift(sigma; v, a)        ← sigma 经过一次单调重映射，与 shift 有关
```

实测同一组 sigma 下 `(12,2)` / `(12,3)` / `(6,2)`：`t_v` 逐位相同（0.1 / 0.5 / 0.8），
`t_a` 各不相同（σ=0.9 时分别 0.400 / 0.308 / 0.250）。所以：

- **改 shift**：8 步的表里 17 行，视频那 8 行 + 固定的 0.999 全部继续有效，
  只缺 8 个音频行（≈158 MiB）——补录一次即可，不是换一张表；
- **改步数 / 调度器 / 自定义 sigmas**：sigma 网格整个换掉，t 值几乎两两不同
  （只有最后一个 `t=1` 会重合），等于整组重录；旧行仍然有效，只是不够用。

参考图/首尾帧/参考音频同理：**只影响"有没有这一类段"，不影响"有几张、是哪张"**。
一份 `ref_img` 和九份 `ref_img` 算出的 t 集合完全相同（都只多一个 `0.999`），
换成别的图更是没有任何区别 —— 因为那一行是 `adaln_proj(time_embedder(0.999))`，
跟图本身无关。

#### 这两行是自动补的（`fill_reference_row`，默认开）

录表时，节点会在**第一次 forward 里**把这一步的 t 和 `0.999`（参考音频再加 `1.0`）
拼成一个批，走**同一套模型模块**算出那一行，然后把多出来的行丢掉：

- 采样结果**一个字节都没变** —— 模型的输入输出完全不受影响，那次出片照常能用；
- 那一行的值与"真的带参考段跑一次"完全一致，因为就是同一个批量、同一套权重算的；
- 所以一张表同时覆盖纯文生视频和参考/首尾帧流程，不用先想好以后用哪种。

关掉这个开关就回到旧行为：纯 t2v 录出来的表里没有 `0.999`，将来第一次上参考会在
**采样前**报 t-set mismatch（不会出废片），再录一次即可。

为什么必须"借模型自己算"而不是事后补：每一步 forward 只算该步出现的 2–3 个不同 t
（video 一个、audio 一个，有参考再加 0.999）。实测（fp16，K=2688）：

- 在批量里**多加一行**不改变其他行：M=2→3，`max diff 0.0`；
- 但把那一行**单独**丢进去算（M=1）会走 GEMV 而不是 GEMM，与批量里的值差
  `0.001953125`（2 个 fp16 ulp）。

所以插件走的是"同一批多算一行再丢掉"，既不动 layout、也不碰 payload。

换步数 / 调度器 / shift 也不会让已有的行失效，只是新 schedule 可能需要表里没有的 t
—— 在教师上再录一次就会**追加**到同一张文件（§3.3），不是换一张表。

由此：

1. 表是 `t -> 调制向量` 的查找表，按 t 索引，不按调用顺序；
2. 同一个模型跑几段采样（二采、refine、hires、SelfLift 高低段）**共用一张表**，
   行数是各段 t 的并集；
3. 换模型 / 换 AdaLN LoRA / 换计算精度 → 另一张表；
4. 普通 LoRA（注意力、MLP、norm）不影响调制，**一张表通吃**。

---

## 2. 节点与参数

### `H3 AdaLN`（日常只用这一个）

| 输入 | 说明 |
|---|---|
| `model` | 必需。录制时挂只读 hook，应用时挂注入 hook |
| `mode` | `apply`（默认）/ `record` / `off` |
| `for_model` | **录制时选一次**：从下拉里选这张表要给哪个剪枝基座用，节点只读它的 AdaLN 张量算指纹并写进表头。没有 `(auto)` 模式 —— 目录里有多个相似基座时只有你知道要用哪个，节点不扫描、不记忆、不猜测。选择会随 workflow 保存 |
| `table` | 默认 `(auto)`，按表头指纹确定性查找；只有想临时强制用某张表时才动它 |
| `table_in` | 可选连线。同 prompt 做 A/B 时接上一条链录出的表 |
| `after` | 可选连线。**仅排序**，同 prompt 录→用时接另一条链的 LATENT |

输出：`model`、`table`、`report`（接一个文本显示节点就能看）。

### `H3 AdaLN Hybrid`（只有 bX-49 混合模型用）

`model`、`table_a`、`table_b`、`split`。合并结果会落盘并注入当前模型。
`final_layer` 固定取 A（bX-49 就是这么拼的）。

### `H3 Latent Compare`（验证用）

两个 LATENT → 报告 `equal / max_abs / PSNR`，`PASS` 表示逐位一致。

---

## 3. 表：名字、身份、行合并

### 3.1 名字是算出来的

身份 = `模型指纹 + AdaLN LoRA 指纹 + 计算精度`。

- 模型指纹取 block 0 / 中间块 / 最后一块的 `adaln_proj` 权重头 + `final_layer`，
  所以 `fl2va full`、`ref2va full`、`b25-49 full` 互不相同；
- 只有 `time_embedder` / `adaln_proj` 上的 LoRA 会改变身份（104 个 LoRA 里只有
  17 个属于这类，都是加速/蒸馏类）；
- 精度进身份：bf16 录的表和 fp32 录的表是两张表，不会互相污染。

文件名：

```
h3_full_{模型指纹前6}_{精度}[_L{AdaLN指纹前4}].safetensors
例：h3_full_001c09_bf16.safetensors
    h3_full_001c09_bf16_Le3b0.safetensors
```

表放在 `ComfyUI/models/h3_exact_adaln/`（已注册为 model 文件夹，
平台的文件浏览器能直接看到）。

**改名安全**：身份写在 safetensors header 里，节点靠 header 找文件，
改名后下次录制仍然合并到同一个文件，不会另起一张。

### 3.2 用哪张表：录表时声明归属，应用时按指纹找

表属于**全量教师**，而 apply 侧只有**剪枝基座**，基座里没有字段记录它从哪个教师
剪出来（b25-49 基座的 `adaln_t_table` 用的就是 fl2va 的）。所以「这张表给谁用」
只能在录制时由你声明一次：

```
mode=record, for_model = <你准备使用的剪枝基座>
```

节点只读该检查点的 AdaLN 张量（曲线表 + 4 个块的权重首行 + bias，几 KB，
不加载 20GB 模型）算出指纹，写进表头：

```
target_base : 87012ca41991        # 基座指纹
target_name : minimax_h3_hybrid_..._b25-49-int8.safetensors
```

**没有 `(auto)` 模式，也没有记忆文件**。目录里放着几个相似基座时，节点无法知道
你要用哪个；扫描 + 猜测只会静默产出一张归属错误的表（形状、t 集合都对，
只有数值错）。所以这一步是显式选择 —— 值随 workflow 保存，只需选一次。

应用时 `table=(auto)` 的顺序：

1. `table` 里指定了文件 → 用它（**仅本次**，不记忆）；
2. 表头 `target_base` 等于当前模型指纹、且只有一张 → 用它；
3. 其余 → 报错（无归属的表不会被自动使用，也没有「最近一张」这种回退）。

于是：

- **选一次，之后零交互**：录一次 → 换基座 → apply，什么都不用选；
- **两个模型**：一采 b25-49、二采 ref2va 各录一次，各自的基座指纹不同，
  两条链各自命中自己的表；
- **归属写在表里**：表拷到别的机器照样能被认出来，插件没有任何外部状态文件。

为什么不用「最近一张」：那是时间维度的猜测，多 pass 必然踩空，
而且选错表是**静默**的（形状、t 集合都对，只有数值错）。

**表改名 / 搬目录**：完全不影响，查找只看表头，不看文件名。

**同一个教师换 schedule 重录**：还是同一个文件，行会合并（§3.3），归属不变。

### 3.3 同一个身份 → 行是累加的

| 情况 | 行为 |
|---|---|
| 同样的 schedule 再录一次 | `unchanged`，**不重写文件** |
| 换了个 schedule（或者加了步数） | `merged`，新 t 行**追加**进去，旧行保留 |
| 同一 t 出现了不同的调制值 | **报错，一个字节都不写** |

第三种只会发生在「模型、AdaLN LoRA、精度真的变了」。普通 LoRA 换了不会触发，
因为身份里没有它。报错长这样：

```
H3AdaLN: this run produced different AdaLN values for t=0.9230769 than
h3_full_001c09_bf16.safetensors (max delta 3.1e-02 on blocks.17). The model,
the AdaLN LoRA or the compute dtype must have changed. Nothing was written;
delete ... if you really want a fresh table.
```

这条正是「不用再按 pass 分表」的原因：录两次不会互相覆盖一半 t。

### 3.4 表的体积

一行 ≈ 19.7 MiB（51 投影 × 96768 × 4B）。17 行 ≈ 315 MiB。
每次录新 schedule 是**加行**，不是换文件，多尺度流程注意磁盘。

---

## 4. 典型场景

### 4.1 标准：全量录 → 剪枝用

1. 全量教师链，在「LoRA 之后、采样器之前」插 `H3 AdaLN`，`mode=record`，
   跑一次你**实际要用**的流程（同样的步数/调度器/shift；参照图/关键帧的有无也要一致）。
   分辨率和帧数不影响表，按真实出片尺寸跑只是为了顺手验证。
   在 `for_model` 里选中准备使用的剪枝基座（随 workflow 保存，只设一次），
   它的指纹会写进表头。
2. 跑完看控制台：`H3AdaLN: saved <path> (17 new rows, 17 total, 315 MiB)`。
   出现这一行才算成功；`unchanged` 说明这张表已经有这些 t 了。
3. 换成剪枝基座，`mode=apply`，`table` 保持 `(auto)`，直接跑 ——
   表头里写着这个基座的指纹，不需要选。

**插入位置**：模型同时喂调度器和采样器的工作流（导演台、YZ 系列），插在
**分叉之前**，让两条下游拿到同一个 patched MODEL。

### 4.2 换 schedule 后报 t-set mismatch

```
H3AdaLN: cannot map (17, 8) rows to table rows.
missing from table (3): 0.9230769, 0.9411765, 0.9600000
```

不用改表、不用删表：把 `mode` 切回 `record`，用**新的** schedule 在教师上跑一次，
缺的行会追加进同一张表。然后再切回 `apply`。

**不会"用错表"**：步数 / 调度器 / shift 都**不是表身份的一部分**（身份只有
基座指纹 + AdaLN-LoRA 指纹 + 精度），所以换这些参数不会指向另一张表 ——
同一模型 + 同一 AdaLN LoRA 永远只有一张表，重录是**合并**。运行时的行为只有两种：
覆盖到了就用（正确），没覆盖就在**第一次 forward** 抛错（`max_dist` 与 `tol` 差着
好几个数量级，不存在"差不多就匹配"）。

### 4.3 同一工作流里两个不同模型（二采换模型 / 高低模）

两个 `H3 AdaLN`，各自挂在各自的模型链上，各自 `mode=record`。
两份表身份不同、文件名不同，都会落盘，互不干扰。

应用侧**不用选表**：两个基座的指纹不同，各自的表在录制时就声明了归属，
两条链各自命中自己的那张，同一个工作流里不会互相串。
万一某张表没声明归属、而目录里又有多张候选，节点会报错让你在 `table` 里选一次。

同一个模型的**同一段**采样只挂一个 record 节点（模型侧 hook 只能有一个活的）。
一张表本来就覆盖多段；如果同一模型链上真的插了两个 record 节点，后一个会接管，
前一个什么都存不下来，日志里会有
`another record node ... the other saves nothing`。

### 4.4 一个 prompt 里既录又用（A/B 验证）

只有这种图需要连线：

```
A 链: 教师 → H3AdaLN(mode=record) → 采样器A ──┐
                                              ├─→ H3LatentCompare
B 链: 剪枝 → H3AdaLN(mode=apply, table_in=A的表) → 采样器B
                      ↑ after ← 采样器A 的 LATENT
```

`after` 只负责让 B 等 A 采样完；表本身用 `table_in` 直连。
日常用法不需要 `table_in`，也不需要 `after`。

### 4.5 bX-49 混合模型

**推荐**：直接在有全量 bX-49 教师上 `mode=record`。一次录制，不需要合并。
已发布的 b5/b20/b25/b30/b45 全量教师：`binglingzhimeng/MiniMax-H3-b25-49-full`。

**没有全量混合教师时**：`H3 AdaLN Hybrid`（`split=25`）可以合并两张表，
但**不精确**：两个官方教师的 `time_embedder` 不同（`proj_out.weight` 最多差
0.104），直接录 Ref2VA 会把 donor 自己的嵌入带进尾部，25 个投影偏 0.09~0.19。
v1 的 `time_embedder_from` 补救节点已删除，所以唯一的精确做法还是拿到全量
混合教师直接录。详见 README「Hybrid (spliced) teachers」。

### 4.6 录表建议：把尺寸压到最小

表只跟 **t 集合**有关，跟分辨率、帧数无关（§1）。而录表必须在全量教师上跑一次
完整采样，34GB 教师每一步都很贵，所以录表那次应该尽量小。

ComfyUI 0.44 的 H3 节点（`comfy_extras/nodes_minimax_h3.py`）参数下限：

| 参数 | 节点允许 | 实际对齐 |
|---|---|---|
| `width` / `height` | `min=32`，`step=32`，`max=16384` | 会被 `/16` 成 latent，再按 `patch_size=(1,2,2)` 切成 token，所以 32 的倍数才整除 |
| `length` | `min=5`，`step=17`，`max=3600` | `align_frame_count` 会向上对齐到 **17k+5**（5 / 22 / 39 / 56 …）；5 帧时 `video_latent_t=2`、`audio_t=8` |

本机实测 `_empty_av_latent` 能构造的合法形状：

```
 32x 32,  5 帧 -> (24, 2, 2, 2)     节点允许的理论下限，空间只剩 1x1 token
256x128,  5 帧 -> (24, 2, 8,16)     建议值
320x192,  5 帧 -> (24, 2,12,20)     更稳一点
384x224, 22 帧 -> (24, 7,14,24)     第二档帧数
```

**建议录表用 `256x128` / `320x192` + 5 帧**：接近理论下限但仍有足够的空间 token，
不会踩极小尺寸下 RoPE / 注意力的边界情况；教师前向一步只要几秒。

必须与实际使用**保持一致**的只有这些（它们决定 t）：

- 步数、调度器、自定义 sigmas
- video / audio shift
- 是否有关键帧 / 参考图 / 参考音频（`fill_reference_row` 默认已把 0.999、1.0 补齐，见 §1）

反过来，这些可以随便改、不影响表：分辨率、帧数、seed、采样器实现、提示词、CFG。

> 注意：`length` 填 5 以外的值会被静默向上对齐（例如填 6 → 22 帧），
> 表是按对齐后的帧数对应的 t 录的；只要录和用填同一个数就没问题。

---

## 5. 怎么判断「对了」

### 5.1 看注入日志，不看最终画面

每次 prompt 结束时，控制台会打印：

```
H3AdaLN: apply stats
mode             : strict
injection mode   : curve-coordinate (8-wide)
t rows in table  : 17
projections      : 51
exact matches    : 102
approx matches   : 0 (max_dist=0.000e+00)
match failures   : 0
native steps     : 0
```

判据：

- `exact matches` = `51 × 采样步数`（节点在报告里写 `armed: 51 projections`）；
- `native steps = 0`，`match failures = 0`；
- 出现 `match failures > 0` 会直接抛错并列出缺失的 t。

**不要用最终 latent 的 PSNR 判断表是否生效**：转换点之后的非 AdaLN 路径会污染
结果（§7.6）。

### 5.2 每次 PASS 都要有负对照

同一个工作流把 `mode` 设成 `off` 再跑一次，结果必须**明显不同**（典型 14~30 dB）。
相同就是缓存或占位，见 §7.1。

### 5.3 端到端逐位对照

`H3LatentCompare`，或 `H3LatentSave` + 离线 `torch.equal`。
A 链 = 全量教师原生，B 链 = 剪枝 + 表，两链共享 conditioning / noise / sigmas / seed。

已通过的矩阵（旧节点时代，引擎未变）：

| 工作流 | 结果 |
|---|---|
| 05 双链（标准采样器） | PASS bitwise |
| DMAD8 步（标准采样器） | PASS bitwise |
| 导演台一采+二采（两个不同模型、自定义 sigmas） | PASS bitwise |
| YZ 终极篇（SelfLift） | 表有效但不 bitwise，见 §7.6 |

### 5.4 本机（2026-11-09）实测到什么程度

ComfyUI 0.44.2 / torch 2.12.1 / GTX 1660 SUPER 6GB / 16GB 内存：

- 真实 `MiniMaxH3Model` 小尺寸数值回归 **21/21 PASS**，含 record→apply 逐位一致
  与 `mode=off` 负对照不通过；
- 真实 ComfyUI 进程内 **7/7 PASS**：真实 `ModelPatcher.cleanup()` 触发落盘、
  换 schedule 合并、**文件/模块两端指纹逐字节一致**、按指纹找表、apply 挂载/卸载、
  off 清理、hybrid 合并；
- 真实 `b25-49 int8` 检查点：51 个投影 `[96768,8]` F16、`final_layer [10752,8]`、
  `adaln_t_table [1025,8]`、**没有** `time_embedder`。用它自己的行在真实宽度上
  注入，51/51 `torch.equal`，`max_dist=0`；5 行表 92.5 MiB。

**没验**：真实 20GB 模型的完整出图（本机没有全量教师，6GB 显存只能跑极低分辨率），
以及真机队列里 `ON_CLEANUP` 的触发时机。

---

## 6. 现在的节点与工作流

| 节点 | 用途 |
|---|---|
| `H3 AdaLN` | `record` / `apply` / `off`，日常只需要这一个 |
| `H3 AdaLN Hybrid` | bX-49 两表合并（见 §4.5，不精确，优先用全量混合教师） |
| `H3 Latent Compare` | A/B 逐位对照 |

v1 的 Record / Bake / Save / Load / Merge / Apply / LatentSave 节点和离线
Bake 引擎已删除，依赖它们的 01~08 工作流也一并删除。现在 `workflows/` 里是
两份 **API 格式** 的 prompt（不是 UI 工作流导出）：

- `v2_record_apply_ab.json` —— 单 prompt 双链 A/B，录制 + 注入 + 逐位比较；
- `v2_apply_auto.json` —— 跨 prompt 自动查表，只有剪枝基座。

它们提交到 ComfyUI 的 `/prompt` 接口即可运行，云实例实测过两份都能 PASS。

**旧表**：v1 表没有 `target_base`，插件不会自动使用（避免静默用错教师），
也不会再扫 `output/h3_adaln/`。要用就重录，并指定 `for_model`。

---

## 7. 陷阱清单

### 7.1 ComfyUI 缓存制造假 PASS

改模型 / 改表 / 改 mode 后，如果下游采样器命中缓存，注入根本没发生。
**症状**：日志里没有 `first injection confirmed`，`psnr=inf` 但负对照也 `inf`。
**对策**：换 seed、改一个无关 widget、或重启 ComfyUI。

### 7.2 导演台的 IMAGE 输出是全零占位

`H3SeamlessChainSampler` 在「自动保存=分段」模式下，`图像` 输出是全零张量；
真实 latent 在 `output/h3_projects/<存档目录>/seg_000.pt`（`video` / `audio` 两个张量）。
它按存档目录缓存分段、有全局 `h3chain_state.json`，换模型不重算。
**对策**：全新目录名 + 删 `h3chain_state.json` + 重启，再比对 `seg_000.pt`。

### 7.3 没有“最近一张”这种东西了

`_last_table.json` 和 `_bindings.json` 都删掉了。旧的 `(latest)` 会随最后一次
落盘漂移，多 pass 时尤其容易串台；现在归属写在表头（`target_base`，§3.2），
应用侧既没有“时间”维度，也没有外部状态。

如果发现某张表声明错了基座：重新录一次并把 `for_model` 指向正确的基座；
或者临时用 `table` 强制指定。

### 7.4 UI 工作流 ≠ API prompt

`GetNode/SetNode` 是前端虚拟总线，不在 `/object_info` 里；动态 COMBO 节点的
`widgets_values` 与 API 输入名不一一对应。低开销模拟时按 `input_order` 手工
构造最小采样主干更快；`MiniMaxH3ReferenceToVideo` 的 `ref_*` 全是 optional。

### 7.5 表相关的静默失败（都已修，但要知道症状）

- 部分替换 `time_embedder` 会混两个嵌入 → v1 曾硬报错；该能力已随旧节点删除。
- 上一次的模块替换残留到下一次录制 → v1 进节点先撤销；该能力已删除。
- 降级到 weight patch 会被降到 bf16（非 bitwise）→ v1 报告会明写；已删除。
- full→pruned 主路径被「模型指纹相等」检查误杀 → 现在只在 full 目标上比较。
- **v2 新增**：同一 t 出现两种调制值 → 拒绝写盘，不会静默保留其中一份。

### 7.6 SelfLift 的 30 dB 残差（不是 bug，别再改插件）

SelfLift 在两段之间插入 clean endpoint → 学习式 `LatentResizer3D` 上采样
→ 重加噪 → 高分辨率段。单链逐阶段 dump 证明：转换点前全部逐位一致，分叉只在
转换点之后；backbone / `comfy_quant` / `weight_scale` 在两个 checkpoint 间也
逐位一致，表 102/102 精确注入、0 回退、0 native 步。

剩下的差异来自 34GB 全量模型（offload / 动态权重加载）与 20GB 剪枝模型
（可全驻留）执行路径不同，在 continuation sampling 下被放大。
试过的插件侧改法（按目标原生 fp32 注入、关掉 sigma wrapper）都无效或更差。
**失真的是「全量 vs 剪枝双链逐位一致」这个验收指标本身**；实际出片只跑一条
剪枝链，表正常工作。

### 7.7 不要用 Bake 期待 bitwise

离线路径看不到采样时 CUDA kernel 的真实数值，恒有约 1 ulp 偏差，
8 步后放大到 latent `mean_abs ≈ 0.145`（PSNR ≈ 31 dB）。
所以它已经不在菜单里了。

### 7.8 `out_dtype` 进身份

bf16 教师录的表、fp32 教师录的表是两张表。换精度不会被当成同一张合并，
也不会互相覆盖——但 apply 到精度不符的模型上会打印
`WARNING: ... the result will NOT be bitwise`。

### 7.9 声明错了会一直错（唯一剩下的静默错误）

apply 侧能查的只有「基座指纹 + AdaLN LoRA 指纹」。**基座本身认不出教师**：
它的 `adaln_proj` 是 `[96768, 8]`，全量教师是 `[96768, 2688]`，形状天然不同，
「相等检查」会把主路径全部误杀（v1 踩过这个坑）。

所以：给 ref2va 的剪枝基座声明的却是一张 b25-49 的表，t 全对、注入 102/102、
不报错，数字是错的。好在声明是**录制时的一次显式动作**，写死在表头里，
不会被后来的录制悄悄改掉。

判断依据：`report` 第一行会写清楚这张表的来路
（`declared for this model` / `only table without a target` / `picked`），
加上表头里的 `target_name`。

### 7.10 权重 fp16 / 计算 bf16，指纹要能跨过去

录表侧用 `base_fingerprint_from_file()` 读的是检查点里的 **F16** 权重；
apply 侧 `base_fingerprint(dm)` 读的是 ComfyUI 加载后的模块 —— 在
`int8_convrot` 检查点上日志会写 `model weight dtype torch.bfloat16`。
fp16→bf16 会丢尾数，两侧字节流不同，结果就是**明明录了表却报
「N 张表都没声明归属」**。

插件现在把基座指纹的每个分量先舍入到 bf16 精度再哈希，两侧一致；
教师身份指纹不受影响。这个坑在 2026-11-09 的云实例实测里首次暴露并修复
（见 HANDOFF §0.1）。症状回顾：换过精度/量化格式后如果自动查找突然失效，
先看日志里的权重 dtype，再看表头 `target_base` 与当前模型指纹是否一致。

另外注意：**`for_model` 没有 `(auto)` 模式**。目录里有多个剪枝基座时，
必须在下拉里显式选一个；不选的话表头不写 `target_base`，`table=(auto)`
会拒绝使用这张表（只能在 `table` 里手动指定）。云实例上就是多个基座的情况，
实测也用的显式指定。

---

## 8. FAQ

**Q：录完 report 里说表是空的？**
A：`report` 是节点执行时生成的，那时候还没采样。看 prompt 结束时控制台的
`H3AdaLN: saved / merged / unchanged`。

**Q：一直没出现 `saved`？**
A：三种可能：采样没跑（看有没有 `first rows captured`）；`ON_CLEANUP` 被别的
插件清掉；同一 t 出现冲突（日志有 `refusing to save`）。

**Q：下拉框里没有刚录的表？**
A：不影响。`table=(auto)` 看的是表头里的基座指纹，不依赖下拉框刷新。
想让它立刻出现在列表里就刷新一下页面（重新拉 `/object_info`）。

**Q：报 `no table for this model`？**
A：目录里没有声明给这个基座的表。要么去录一张（`mode=record` + `for_model`
指向这个基座），要么这次就是不打算用表，把 `mode` 设成 `off`。

**Q：报 `tables fit this model and none of them declares a target`？**
A：有几张表都没写归属。在 `table` 里指定一次，或者重录时补上 `for_model`；
细节见 §3.2。

**Q：换 LoRA 后要重录吗？**
A：风格/动作类不用（84% 的 H3 LoRA 不碰 AdaLN）。加速/蒸馏类
（turbo、Acc、dmd、lms）碰 `adaln_proj`，要重录；表现是文件名多了 `_Lxxxx`
或者 apply 直接报 AdaLN 指纹不符。

**Q：表放哪、怎么分享？**
A：`ComfyUI/models/h3_exact_adaln/`，普通 `.safetensors`，拷给对方放进同名目录即可。

**Q：一行多大？**
A：约 19.7 MiB。17 行 ≈ 315 MiB。

---

## 9. 相关文档

- `README.md` —— 精确性契约、实测证据、bX-49 混合模型的来龙去脉
- `docs/unified-node.md` —— v2 的设计不变式：身份、行合并、为什么取消 strict/fallback
- `HANDOFF.md` —— 全部实测证据、根因与换机速查
- `workflows/v2_record_apply_ab.json` —— 双链 bitwise 对照模板
- `workflows/v2_apply_auto.json` —— 跨 prompt 自动查表模板
