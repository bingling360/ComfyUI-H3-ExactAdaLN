# H3 AdaLN 统一节点使用教程

> 适用版本：`ef8c2fc` 及之后。节点名 `H3AdaLN`，显示名 **H3 AdaLN (record / apply / merge)**。
> 本文同时收录 2026-11-08/09 两天在云端实例上做的全部实测结论与踩过的坑，
> 建议第一次使用前把 §0 和 §8 先读一遍。

---

## 0. 经验教训总结（先看这个）

### 0.1 关于表本身

1. **表是 `t -> modulation` 的纯函数**：`modulation = adaln_proj(time_embedder(t))`。
   式子里没有 pass、latent、分辨率、seed、conditioning。所以同一模型的多段采样
   （二采、refine、hires、SelfLift 高低段）天然共用一张表，行数是各段 t 的并集。
   **不要按 pass 分表** —— 分了反而每张都只覆盖一半 t，另一段必然 t-set mismatch。
2. **分表的维度是「模型 + AdaLN LoRA」，不是 pass**。只有两条链用了不同的
   checkpoint / AdaLN LoRA 时，才需要两个节点、两张表，用 `pass_tag` 区分。
   实测一采 b25-49 / 二采 ref2va 两张表指纹不同、t 集合不重叠、交叉使用被 strict 拒绝。
3. **84% 的 H3 LoRA 不碰 AdaLN**（104 个样本里 87 个），一张表通吃；只有
   加速 / 蒸馏类（turbo、Acc、dmd、lms）才需要重录。非 AdaLN 指纹不匹配
   **不该**让 strict 失败 —— 这个误判已修。
4. **录制 donor 教师时必须换 `time_embedder`**。FL2VA 与 Ref2VA 的
   `time_embedder.proj_out.weight` 最大差 0.104，直接录 donor 会把 donor 的
   嵌入烤进表，尾部偏 0.09~0.19。而且必须整体替换模块（object patch），
   weight patch 会被降到 bf16 造成 ~1e-1 偏差；hook 也要挂到替换后的模块上。
5. **全量教师与剪枝基座的 `adaln_proj` 形状本来就不同**（2688 宽 vs 8 宽），
   所以「模型指纹相等」检查会误杀主路径。现在只在目标仍是 full 模型时比较；
   pruned 基座上表会整体替换 curve，其自身 AdaLN 权重无关紧要。
6. **表的 dtype 不要乱改**：内部 fp32 保存，注入时按全量教师原生 bf16 降精度。
   实测改成按目标模块原生 fp32 注入，结果从 17.76 dB 掉到 10.21 dB。
7. **每行约 19.7 MiB**（51 投影 × 96768 × 4B）。15 行 ~296 MiB，
   上百行的 SelfLift 多尺度流程会到 GB 级，注意磁盘。

### 0.2 关于验证方法（最容易出错的部分）

8. **ComfyUI 缓存会制造假 PASS**。改模型 / 改表 / 改模式，如果节点返回的
   MODEL 对象引用没变，下游采样器会命中缓存，注入日志里
   `first injection confirmed` 计数为 0。对策：换 seed、改无关 widget、
   或重启 ComfyUI 清缓存。
9. **每次 PASS 都要有负对照**：把 `mode=off`（不加表）再跑一遍，必须 FAIL，
   否则 PASS 不可信。今天至少三次「逐位一致」是假的，全靠负对照抓出来。
10. **判断表是否生效看注入日志，不是最终 latent 的 PSNR**。
    `exact matches` / `match failures` / `native steps` 才是真凭据；
    最终 latent 会被转换点之后的非 AdaLN 路径污染（见 §8.6）。
11. **导演台（SequenceForge）采样器的 `图像` 输出是全零占位张量**，真实 latent
    写在 `output/h3_projects/<存档目录>/seg_000.pt`；而且它按存档目录缓存分段，
    换模型不重算。必须：全新目录名 + 删 `h3chain_state.json` + 重启清缓存，
    再比对 `seg_000.pt` 里的 `video` / `audio` 两个张量。
12. **UI 工作流 ≠ API prompt**：`GetNode/SetNode` 是前端虚拟总线，不在
    `/object_info` 里；动态 COMBO 节点（`BlockSparseAttention`）的
    `widgets_values` 与 API 输入名不一一对应。低开销模拟时，按 `input_order`
    手工构造最小采样主干，比写通用转换器快且可靠；参考媒体缺失时利用
    `MiniMaxH3ReferenceToVideo` 的 `ref_*` 全是 optional 这一点裁掉。
13. **两条对照链必须共享 conditioning / noise / sigmas / seed**，否则比较无意义。
14. **改第三方插件源码做诊断必须备份并恢复**，且每次改动都要重启 ComfyUI。

### 0.3 关于边界

15. **SelfLift 的残差不是插件 bug**。单链独立运行逐阶段 dump 证明：
    转换点 `z0_low` / `z_lat` / `z0_high`、重加噪后的 `resume_latent`
    全部逐位一致，分叉只发生在高段模型 forward；而 backbone 权重、
    `comfy_quant`、`weight_scale` 在两个 checkpoint 之间也全部逐位一致，
    表注入 102/102 精确、0 回退。剩下的差异来自 ComfyUI 对 34GB 全量模型
    （offload / 动态权重加载）与 20GB 剪枝模型（可全驻留）的不同执行路径。
    试过的两种插件侧改法（改注入 dtype、关 sigma wrapper）都无效或更差。
16. **失真的是「全量 vs 剪枝双链逐位一致」这个验收指标本身**，因为参照链
    受 offload 影响。实际出片只跑一条剪枝链，表正常工作。
17. **`_last_table.json` 在多个 pass 同时录制时会被后写的覆盖**，
    多 pass 应用必须用 `table_text` 显式指定，不能靠 `auto` / `last`。

---

## 1. 这个节点解决什么问题

MiniMax-H3 的 **pruned / curve 版** 模型把 AdaLN 分支折成 8 维曲线
（`adaln_t_table [1025,8]` + `[96768,8]` 投影），省掉约 13B 参数，代价是：

- 加 LoRA 后，ComfyUI 会**丢掉** LoRA 里 `time_embedder` / 2688 宽
  `adaln_proj` 的键，导致调制量偏离教师模型；
- 偏离只有 1 个 bf16 ulp，但经过 8 步扩散会放大成 latent 偏移
  （实测 mean_abs ≈ 0.145，PSNR ≈ 31 dB）。

插件的做法：在**全量教师**上、用真实采样过程录下每个 t 的原生调制量，
存成表；在 **pruned 基座**上把曲线输出整体替换成表里的行。
`H3AdaLN` 把原来的 Record / Save / Load / Apply / Merge 五个节点合成一个，
用 `mode` 切换。

### 与旧节点的关系（老的都保留，可以混用）

| 旧节点 | 新节点 |
|---|---|
| `H3 AdaLN Record` | `mode=record` |
| `H3 Exact AdaLN Save` | 不需要，`ON_CLEANUP` 自动落盘 |
| `H3 Exact AdaLN Load` | 下拉栏 / `table_text` |
| `H3 Exact AdaLN Apply` | `mode=apply` |
| `H3 AdaLN Merge` | `mode=merge`（只存盘）/ `mode=hybrid`（存盘并注入） |

---

## 2. 安装与位置

节点注册在 `__init__.py` 里，用 `try/except` 包住 —— 即使导入失败也不会
拖垮原来 8 个稳定节点。导入时会创建表目录并注册成 ComfyUI model folder：

```
ComfyUI/models/h3_exact_adaln/
```

所有表都放这里，包括录制产出、合并产出、手工下载的表。
目录里还会有一个 `_last_table.json` 指针文件（见 §5.2）。

**验证过**：ComfyUI 0.42.5 / torch 2.9.1 / RTX 4080 SUPER 32GB。
其他版本若 `CallbacksMP.ON_CLEANUP` 行为不同，录制可能不落盘，看日志里
`H3AdaLN: saved ...` 是否出现。

**插入位置原则**：`LoRA 链之后 → 采样器之前`。如果同一个 MODEL 同时喂
`BasicScheduler` 和采样器（导演台、YZ 工作流都这样），插在**分叉之前**，
让两条下游都拿到同一个 patched MODEL。

---

## 3. 输入 / 输出一览

### 必需

| 输入 | 类型 | 说明 |
|---|---|---|
| `model` | MODEL | 直通。录制时挂只读 hook，应用时挂注入 hook |
| `mode` | COMBO | `apply` / `record` / `hybrid` / `merge` / `off` |

### 可选

| 输入 | 默认 | 说明 |
|---|---|---|
| `table` | 下拉第一项 | `apply` / `hybrid` / `merge` 用的表 A |
| `table_b` | 下拉第一项 | `hybrid` / `merge` 用的尾部供体表 |
| `table_source` | `auto` | `auto` / `dropdown` / `last`，见 §5.2 |
| `table_text` | 空 | **最高优先级**：文件名 / 唯一子串 / 绝对路径 |
| `table_b_text` | 空 | 同上，给表 B |
| `split` | 25 | 从第几个 block 开始取表 B |
| `final_from` | `a` | `final_layer` 取哪张表 |
| `strictness` | `strict` | `strict` / `fallback`，见 §7.2 |
| `record_name` | 空 | 留空则按标签自动命名；同配置重录覆盖 |
| `model_tag` | 空 | 写进表名，如 `b25-49full-int8` |
| `lora_tag` | 空 | 写进表名，如 `dmad8x1.0` |
| `pass_tag` | 空 | 多模型分表时用，如 `p1` / `p2` / `high` / `low` |
| `scheduler_tag` | 空 | 自由文本；调度器名对模型侧节点不可见，步数与 shift 是实测的 |
| `table_in` | 可选连线 | 同一次运行内接另一条链录出的表（A/B 验证） |
| `after` | 可选连线 | **仅排序**，不参与计算。接另一条链的 LATENT，见 §6.4 |

### 输出

| 输出 | 说明 |
|---|---|
| `model` | 直通的 MODEL |
| `table` | 表对象（录制中是「活表」，采样结束后才有行） |
| `table_name` | 落盘后的文件名（`apply` 时是所用表的路径） |
| `report` | 人读的诊断文本；建议接一个文本显示节点 |

---

## 4. 命名规则

```
{model}__{lora}__[pass__]{scheduler}{steps}__sh{sv}-{sa}__{fp6}.safetensors
```

例：

```
b25-49full-int8__dmad8x1.0__simple8__sh12-2__001c095c7e10.safetensors
b25-49full-int8__dmad8x1.0__p2__simple8__sh12-2__001c095c7e10.safetensors
```

- 每段只保留 `[A-Za-z0-9._-]`，截断到 32 字符；`fp6` 是
  `lora_fingerprint` 前 6 位，它已把 LoRA 文件与强度揉进去，所以同 LoRA 不同强度不会撞名。
- **文件名里不放 t 行数**：同配置重录是覆盖，不会堆出 `__t7` / `__t15` 一堆文件。
  行数写进元数据和 report。

元数据里写全：`model_tag` / `model_fingerprint` / `lora_tag` / `lora_fingerprint` /
`pass_tag` / `scheduler_tag` / `shift_v` / `shift_a` / `t_rows` / `t_set_hash` /
`out_dtype` / `repeat_conflicts` / `max_repeat_delta` / `created`。

---

## 5. 五种模式

### 5.1 `record` —— 录表

```
模型链 ── H3AdaLN(mode=record) ── 采样器
```

- 节点在 `adaln_proj.linear` 和 `time_embedder` 上挂只读 forward hook，
  **表对象在执行时是空的**，采样过程中逐步填行。
- **不需要 Save 节点，也不需要 `after` 连线**：`ON_CLEANUP` 回调在 prompt
  结束时统一落盘。日志里出现 `H3AdaLN: saved <path> (N t rows, X MiB)` 才算成功。
  （若改用旧五节点，仍然需要 `H3 Exact AdaLN Save Table` + `after` 排序。）
- 落盘前会检查：`rows > 0`、`repeat_conflicts == 0`、名字已定。
  `repeat_conflicts > 0` 意味着**同一个 t 出现了两种调制值**（中途换了模型 /
  AdaLN LoRA），会拒绝保存并报出冲突的 t。
- 建议填 `model_tag` / `lora_tag` / `scheduler_tag`，让文件名可读。
  留 `record_name` 为空即自动命名；填了则用填写的名字（自动补 `.safetensors`）。

### 5.2 `apply` —— 用表（含「录完无缝切模式」）

```
模型链 ── H3AdaLN(mode=apply) ── 采样器
```

选表策略 `table_source`：

| 值 | 行为 |
|---|---|
| `auto`（默认） | 下拉项能解析就用它；解析不了退回 `_last_table.json` 指向的最近一次录制；都不可用才报错并列出候选 |
| `dropdown` | 只认下拉项，旧行为，适合固定引用 |
| `last` | 只认最近录制，适合「边调边用」 |

**录完立刻切模式**：录制成功后会写
`models/h3_exact_adaln/_last_table.json`。把同一个节点的 `mode` 从 `record`
切成 `apply`，若下拉项为空或已失效，`auto` 会回落到最近录制，不用手打名字。

⚠️ `auto` 的**优先级是「下拉项能解析就先用它」**。如果表目录里已经有多张表、
且下拉栏仍指向其中一张有效旧表，切模式后要手动在下拉里选刚录的那张，
或直接填 `table_text`；想无条件用最近一次录制就选 `table_source=last`。

`strictness`：`strict`（默认）遇 t-set 不匹配、AdaLN 指纹不匹配、
`expected_teacher` 不匹配会直接报错。非 AdaLN 的 LoRA 指纹不同**不会**报错
（实测跨 LoRA 复用逐位一致）；`fallback` 则在上述硬错误时只告警并继续。

### 5.3 `hybrid` / `merge` —— 混合表

两张表按块拼接：`blocks 0..split-1` 取表 A，`blocks split..49` 取表 B，
`final_layer` 由 `final_from` 决定。Merge 会校验两表的 t 集合、投影集合、
`out_dtype`、LoRA 指纹。

- `merge`：只落盘，不改 MODEL。
- `hybrid`：落盘并把合并表注入当前 MODEL。

> **重要**：老配方「FL 全量表 + Ref 全量表」与「真正的 bX-49 hybrid 教师」
> **不是同一个目标**。FL 和 Ref 的 `time_embedder` 不同，合并表尾部是
> `W_ref @ e_ref(t)`，而真实 hybrid 尾部是 `W_ref @ e_FL(t)`，差 `0.09~0.19`。
> 若没有 34GB 的混合 full 模型，正确做法是录 Ref 表时用
> `H3AdaLNRecord.time_embedder_from` 指向 FL 的 checkpoint（见 §6.3）。

### 5.4 `off` —— 对照

透传 MODEL，但会清掉残留的注入 hook。做负对照必备。

---

## 6. 典型场景配方

### 6.1 标准：全量教师录 → 剪枝基座用

最常用、已在 DMAD8 步工作流上验证 **逐位 PASS**（不加表 30.48 / 29.87 dB）。

1. 全量教师链 → `H3AdaLN(mode=record, model_tag=..., lora_tag=..., scheduler_tag=...)`
   → 采样器，跑一次你**实际要用的** schedule / shifts / 分辨率 / 布局。
2. 把同一个节点的 `mode` 切到 `apply`，确认选中的是刚录的表
   （下拉项若仍指向旧表就手动改选或填 `table_text`，见 §5.2），
   把 UNETLoader 换成 pruned 基座。
3. 看 report：`exact matches = 51 × 步数`、`match failures = 0`、
   `native steps = 0` 就对了。

### 6.2 同一张表给多个 LoRA 用

直接换 LoRA，不用重录。实测：用 DMAD8 录的表，配 turbo 8step LoRA
跑 pruned，对照全量 + turbo 原生：`equal=True / psnr=inf`。
只有 LoRA 含 `adaln_proj` / `time_embedder` 键才需要重录（加速蒸馏类）。

### 6.3 没有混合 full 模型时造 bX-49 的表

1. FL2VA full 上录表 A。
2. Ref2VA full 上录表 B，但 **`time_embedder_from` 是旧节点
   `H3 AdaLN Record` 的输入（统一节点 `H3AdaLN` 暂未暴露它）。用旧节点录 B'，
   再把它接到统一节点的 `table_in`，或直接用现成的
   `workflows/06_b25-49_hybrid.json`（生成器已把 `time_embedder_from` 写进 06 的
   Record 节点）。填 FL2VA 的 checkpoint 文件名 —— 它会在录制前把 FL 的
   `time_embedder` 整体换进 Ref 教师。录出的 B' 与真实 hybrid 教师的表
   **25/25 逐位一致**，合并后 **51/51 全等**。
3. `mode=merge`（或 `hybrid`），`split=25`，`final_from=a`。

`time_embedder_from` 的名字解析：精确文件名 / 唯一子串 / 已存在的绝对路径。
常见报错与含义：

| 报错 | 原因 |
|---|---|
| `contains no time_embedder.* tensors` | 传了 pruned 文件；pruned 没有 time_embedder |
| `does not match this teacher's time_embedder` | 形状/名字对不上，节点会拒绝**部分替换**（混两个嵌入） |
| `carries N LoRA patch(es) on time_embedder` | 模块替换会丢掉该 LoRA 的 delta，需二选一 |
| `does not exist (absolute paths are used as-is)` | 绝对路径写错 |

### 6.4 一个工作流里两条模型链（二采 / 高低采不同模型）

**已验证**：导演台二采工作流，一采 b25-49、二采 ref2va，各自自定义 sigmas，
提示词内嵌在采样器的「导演台状态」JSON 里。每个模型链各插一个 `H3AdaLN`：
一采 `pass_tag=p1` + `record_name=seq_p1`，二采 `pass_tag=p2` + `record_name=seq_p2`。
两个 record 节点在同一 prompt 里都会各自 `ON_CLEANUP` 落盘，表指纹不同、
t 集合不重叠、0 冲突。应用侧用 `table_text` 显式指定各自的表 —— 交叉指定
会被 strict 的 t-set 校验挡住（一采 t=0 不在 p2 表里）。

同 prompt 内「录 → 用」且采样器没有自己的 `after` 接口时（SelfLift、导演台），
把另一条链的 LATENT 接到本节点的 **`after`**，强制拓扑排序，让 `table_in`
直连可靠工作。

### 6.5 复杂工作流插入要点

- **导演台（SequenceForge）**：模型同时喂调度器和采样器时，插在
  `ModelAttentionBackend` 之后（分叉前）。已插好的示例：
  `导演台二采+自定义sigmas (2)_H3AdaLN.json`。
- **YZ 终极篇 / SelfLift**：插在「模型」总线之前（`ModelPreviewOverrideKJ`
  与 `SetNode「模型」` 之间）。已插好的示例：`YZ-MiniMax H3 终极篇-文武双全_H3AdaLN.json`。
- **DMAD8 步**：插在 `ModelAttentionBackend` 之后，一采链即可。
  已插好的示例：`DMAD8步-文戏专用_H3AdaLN.json`。

三份「已插入节点」的示例工作流是在本机生成的（含用户私有提示词与素材引用，
故未提交进仓库）。需要时在对应工作流里按上述位置插入一个或两个 `H3AdaLN`，
填入 `table_text` 与 `pass_tag` 即可。

---

## 7. 怎么判断「对了」

### 7.1 正确的判据

看 Apply 的 report / 日志：

```
exact matches    : 102
approx matches   : 0 (max_dist=0.000e+00)
match failures   : 0
native steps     : 0
```

- `exact matches` 应等于 `51 × 采样步数`（SelfLift 两段时是 `51 × 2`）。
- `native steps > 0` 说明有步没注进去，结果不可信。
- `match failures > 0` + strict 会直接报错，列出缺失的 t 值。

### 7.2 负对照

同一个工作流把 `mode` 设成 `off` 再跑一次，与加表的结果比较：
**必须明显不同**（典型 14~30 dB）。相同就是缓存或输出占位，见 §8.1 / §8.2。

### 7.3 端到端 bitwise 对照

用 `H3LatentCompare` 或 `H3LatentSave` + 离线 `torch.equal`：
A 链 = 全量教师原生，B 链 = pruned + 表，两链共享
conditioning / noise / sigmas / seed。已通过的矩阵：

| 工作流 | 结果 |
|---|---|
| 05 双链（标准采样器） | PASS bitwise |
| DMAD8 步（标准采样器） | PASS bitwise |
| 导演台一采+二采（两个不同模型、自定义 sigmas） | PASS bitwise |
| YZ 终极篇（SelfLift） | 表有效但不 bitwise，见 §8.6 |

---

## 8. 陷阱清单（按今天踩的顺序）

### 8.1 ComfyUI 缓存制造假 PASS
改模型 / 改表 / 改 mode 后，若下游采样器命中缓存，注入根本没发生。
**症状**：`first injection confirmed` 日志计数 0，`psnr=inf` 但负对照也 `inf`。
**对策**：换 `seed`、改无关 widget、或重启 ComfyUI。

### 8.2 导演台的 IMAGE 输出是全零占位
`H3SeamlessChainSampler` 在「自动保存=分段」模式下，`图像` 输出是全零张量；
真实 latent 在 `output/h3_projects/<存档目录>/seg_000.pt`（含 `video` / `audio`
两个张量）。而且它按存档目录缓存分段、有全局 `h3chain_state.json`，换模型不重算。
**对策**：全新目录名 + 删 `h3chain_state.json` + 重启，比对 `seg_000.pt`。

### 8.3 多 pass 录制覆盖指针
`_last_table.json` 只记最后一次。两个 pass 同时录制时，应用侧必须
`table_text` 显式指定，不能靠 `auto` / `last`。

### 8.4 UI 工作流 ≠ API prompt
`GetNode/SetNode` 是前端虚拟总线；动态 COMBO 节点 widget 与 API 输入不对应。
低开销模拟时按 `/object_info` 的 `input_order` 手工构造最小采样主干，
`MiniMaxH3ReferenceToVideo` 的 `ref_*` 全是 optional，可直接裁掉缺失的素材节点。

### 8.5 表相关的静默失败（都已修，但要知道症状）
- 部分替换会混两个嵌入 → 现在硬报错。
- 上一次的 `time_embedder` 模块替换残留到下一次录制 → 现在进节点先撤销。
- 降级到 weight patch 会被降到 bf16（非 bitwise）→ report 会明写
  `weight patch - bf16, NOT bitwise`。
- full→pruned 主路径被指纹相等检查误杀 → 现在只对 full 目标比较。

### 8.6 SelfLift 的 30 dB 残差（不是 bug，别再改插件）
SelfLift 在两段间插入 clean endpoint → 学习式 `LatentResizer3D` 上采样器
（或 VAE 往返）→ 重加噪 → 高分辨率段。单链独立运行逐阶段 dump：
转换点前全部逐位一致，分叉只在高段 forward；backbone / `comfy_quant` /
`weight_scale` 在两个 checkpoint 间全部逐位一致，表 102/102 精确注入。
剩下的差异来自 34GB 全量模型（offload）与 20GB 剪枝模型（全驻留）的
执行路径不同，在 continuation sampling 下放大。
**试过且无效/更差的插件侧改法**：按目标原生 fp32 注入（更差）、
关掉 Apply 的 sigma wrapper（无变化）。**不要再往这个方向改。**
失真的是「全量 vs 剪枝双链逐位一致」这个验收指标；实际出片只跑一条剪枝链，
表正常工作。

### 8.7 不要用 Bake 期待 bitwise
`H3ExactAdaLNBake` 是离线路径，看不到采样时 CUDA kernel 的真实数值，
恒有约 1 ulp 偏差。要精确就用 `record`。

### 8.8 表是 (模型, AdaLN-LoRA, sigma 网格, shifts, packed layout) 的函数
任一项变了都要重录。`out_dtype` 在产表时固定，换目标精度会出问题（§0.1-6）。

---

## 9. FAQ

**Q：下拉栏不刷新怎么办？**
A：填 `table_text`（文件名）；或 `table_source=last`（无条件用最近一次录制）；
或 `auto`（下拉项失效时才回落最近录制，见 §5.2）。

**Q：录制完 report 里说表是空的？**
A：采样还没跑完，表在结束时才有行。看日志 `H3AdaLN: saved ...`。
如果一直没出现，检查 `ON_CLEANUP` 是否被别的插件清掉，以及
`repeat_conflicts` 是否 > 0（拒绝保存）。

**Q：报 `t-set mismatch / missing from table`？**
A：当前 schedule / shift 产生了表里没有的 t。用**实际要用的**配置重录，
或用 Bake 的 `extra_t` 补。shift 变了尤其容易触发。

**Q：换 LoRA 后 strict 报错？**
A：先看是不是 AdaLN 类 LoRA（加速蒸馏）。普通风格/动作 LoRA 不会报错；
若报的是 `adaln_lora_fingerprint mismatch`，就重录。

**Q：`model_fingerprint` 相关报错？**
A：只在目标是 full 模型时才会比较。pruned 基座上不会拦截。

**Q：表放哪、怎么分享？**
A：`ComfyUI/models/h3_exact_adaln/`。表是普通 `.safetensors`，
可以直接拷给别人放进同名目录。

**Q：一行多大？**
A：约 19.7 MiB。多尺度流程行数多时注意磁盘。

---

## 10. 完整验证矩阵（2026-11-08/09 实测）

| 项目 | 结论 |
|---|---|
| Cobijada vs 官方源 b25-49 的 51 个 AdaLN 投影 + `adaln_t_table` | 逐位一致；差异只在非 AdaLN 权重（烧进去的解锁 LoRA） |
| FL2VA full 与 Ref2VA full 的 `time_embedder` | **不同**（`proj_out.weight` max 0.104） |
| FL 表 + Ref 表合并 vs 真实 bX-49 教师 | blocks 0-24 逐位一致，25-49 差 0.09~0.19；端到端 22.20 / 31.10 dB |
| donor + `time_embedder_from` 合并 | 25/25 逐位一致，合并后 51/51 全等 |
| 跨非 AdaLN LoRA 复用表 | PASS bitwise（strict 模式下） |
| pruned b25-49 构成 | blocks 0-24 = fl2va_pruned，25-49 = ref2va_pruned，`adaln_t_table` = fl2va_pruned |
| LoRA 扫描（104 个） | 87 个无 AdaLN 键，7 个 curve 形态，10 个 full 形态；加速类全带 AdaLN |
| SelfLift 两段连续采样 | 一张 3 行表覆盖，102/102 精确注入、0 回退、0 native 步 |
| 导演台一采/二采不同模型 | 两张独立表，有表逐位 PASS，无表 14.37 / 16.25 dB，交叉表被 strict 拒绝 |

已发布的 5 份 bX-49 全量教师（b5/b20/b25/b30/b45）：
`binglingzhimeng/MiniMax-H3-b25-49-full`。

---

## 11. 相关文档

- `README.md` —— 项目总览、精确性契约、旧节点与工作流清单
- `docs/unified-node.md` —— 统一节点的设计与不变式推导
- `HANDOFF.md` §11 —— 全部实测证据、根因与换机速查
- `workflows/05_record_strict.json` —— 双链 bitwise 对照模板
- `workflows/06_b25-49_hybrid.json` —— 两教师合并模板（已带 `time_embedder_from`）
