# ComfyUI-H3-ExactAdaLN 交接文档

> **换机后从这里开始。** 本文件自包含，新对话请先完整读完再动手。
> 所有结论都已实测或经源码核对，不要重复排查「已排除项」。
>
> **v2 精简（2026-11-09）**：插件现在只注册三个节点 —— `H3 AdaLN`、
> `H3 AdaLN Hybrid`、`H3 Latent Compare`。v1 的七个单功能节点、离线 Bake
> 引擎、`latent_io.py`、`tools/` 生成器、01~08 工作流**已全部删除**。
> 本文件 §2 / §3 的 v1 清单和 §4 之后的排障记录保留作历史与根因证据，
> 其中出现的旧节点名都已不存在，不要照着去找代码。当前状态见 §0.1。
>
> `README.md` 是给用户看的用法说明，本文件是给下一个接手的开发者看的状态、
> 流程与坑。
>
> **2026-11-11 更新**：`for_model` 的 `(auto)` 模式、模型目录扫描和
> `_default_bases.json` 记忆绑定已移除，录制时必须显式选基座；`table` 侧的
> `(auto)` 指纹查找保留。本文涉及旧行为的段落仅作历史记录，当前用法以
> `README.md` / `docs/NODE_GUIDE.md` 为准。

---

## 0. 换机速查

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/bingling360/ComfyUI-H3-ExactAdaLN`（分支 `main`） |
| 代码状态 | `main` 最新提交；v2 精简版（三节点 + 两份 API prompt） |
| 插件安装位置（实例） | `/root/ComfyUI/custom_nodes/ComfyUI-H3-ExactAdaLN/` |
| 实例环境 | ComfyUI 0.44.0 / PyTorch 2.9.1+cu130 / Python 3.12.3 / RTX 4080 SUPER 32GB |
| 本地开发目录 | 任意 clone 位置（本文档写作时为 `D:/Adaln/ComfyUI-H3-ExactAdaLN/`） |
| 本地 ComfyUI（只为跑测试） | `D:/本地部署文件集合/comfyui/comfy shili/ComfyUI/ComfyUI`，版本 0.41.2，**低于插件要求的 0.42.5，且缺 `comfy_kitchen`，跑不了 `test_exact.py`** |
| 实例 SSH | 主机/端口/密码由用户每次开机后提供，**不要写进仓库**（仓库是公开的） |
| 部署约定 | 用户已授权可直接在实例上 `git pull` + 复制 `workflows/*.json` + 重启；未经授权时只推 GitHub |
| 验收状态 | **已达成**：05 严格验证 PASS（bitwise）、off 对照 FAIL、b25-49 表级逐位 PASS |

**换机后的最小动作**：

```bash
git clone https://github.com/bingling360/ComfyUI-H3-ExactAdaLN.git
cd ComfyUI-H3-ExactAdaLN
python tests/test_simple.py       # 纯 CPU，节点面：身份/命名/行合并/录制落盘
python tests/test_merge.py        # 纯 CPU，验证合并逻辑（需 torch/safetensors/numpy）
```

要跑真实模型，需要 ComfyUI 0.42.5+（实例上是 0.44.0）和「§7 模型清单」里的文件。

### 0.1 节点面已改为 v2（本次改动，先读这段）

节点从「8 个单功能 + 1 个 16 widget 的统一节点」收敛为 **3 个可见节点**：

| 节点 | 输入 | 备注 |
|---|---|---|
| `H3 AdaLN` | `model` / `mode`(apply·record·off) / `for_model` / `fill_reference_row` / `table` / `[table_in]` / `[after]` | 录制、应用、对照 |
| `H3 AdaLN Hybrid` | `model` / `table_a` / `table_b` / `split` | bX-49 合并，`final_from` 固定 a |
| `H3 Latent Compare` | 两个 LATENT | 不变 |

被删掉的 widget 与等价替代：

- `model_tag` / `lora_tag` / `scheduler_tag` / `pass_tag` / `record_name`
  → 文件名由**身份**推导：`h3_{full|pruned}_{模型指纹6}_{精度}[_L{AdaLN指纹4}]`；
- `table_source`(auto/dropdown/last) + `table_text` + `table_b_text`
  → 一个下拉框，默认 `(auto)`；归属由录表时的 `for_model` 写进表头 `target_base`；
- `strictness` → 恒为严格（`fallback` 只会产出「看起来成功、数值是错的」结果）；
- `split` / `final_from` → 移到 `H3 AdaLN Hybrid`。

两个**行为变化**，验收时要一起看：

1. **保存改为按身份合并行**：同 schedule 重录是 `unchanged`（不重写）；
   新 schedule 是 `merged`（追加 t 行）；同 t 不同值则拒绝写盘。
   目的是消灭 `t-set mismatch` 和 `pass_tag` 分表。
2. **归属写进表头**：录表时节点自己认基座（扫模型目录，`adaln_t_table` 有、
   `time_embedder` 无 = 基座；只有一个就直接用），只读它的 AdaLN 张量（几 KB，
   3.7 ms，不加载模型）算出 `target_base`；apply 侧算同一个指纹去查。
   没有 `_last_table.json`，也没有 `_bindings.json` —— 表拷到别的机器照样认得出。
   `for_model` 只在「目录里两个以上基座」时才需要手动指定。
3. **身份含计算精度**：bf16 教师与 fp32 教师录出的是两张表，不会互相合并。
4. **固定行自动补**（`fill_reference_row`，默认开）：录表时把 `t=0.999`（参考音频再加
   `1.0`）追加到该步 t_emb 批里，用模型自己的模块多算一行再丢掉 —— 同批同权重，
   值等于真带参考跑一次，且**不改采样结果**（不动 layout、不碰 payload）。
   关掉则回到旧行为：纯 t2v 表没有 0.999，首次上参考会在采样前报 t-set mismatch。

实现落在 `h3_unified.py`（重写）；`nodes.py` 已瘦身为共享 helper +
`H3LatentCompare`；`latent_io.py`、`bake_engine.py`、`tools/` 和 01~08 工作流
全部删除。`tests/test_simple.py` / `tests/test_runtime.py` 覆盖新节点；
`tests/test_workflows.py` 改为校验两份 API prompt 的结构。

**本机已验（2026-11-09，ComfyUI 0.44.2 / torch 2.12.1 / GTX 1660S 6GB / 16GB RAM）**：

| 项 | 结果 |
|---|---|
| `tests/test_exact.py`（真实 `MiniMaxH3Model`，小尺寸，CPU） | 21/21 PASS，含 record→apply 逐位一致 + `mode=off` 负对照不通过 |
| `tests/test_runtime.py`（真实进程、真实 `ModelPatcher.cleanup()`） | 11/11 PASS：注册目录、文件/模块指纹一致、录制落盘（含 `target_base`）、换 schedule 合并、自动认基座、pin 行逐位一致、缺行提示、apply/清理、hybrid 合并注入 |
| 真实 b25-49 int8 检查点结构 | 曲线形态：`adaln_t_table [1025,8]` F32，无 `time_embedder`，51 个 `adaln_proj` 权重 `[96768,8]` **F16**（不是量化张量），`final_layer` 为 `[10752,8]` |
| 真实权重/真实宽度注入自检 | 51 投影全挂、`torch.equal=True`、`max_dist=0.000e+00`、0 失败；5 行表 92.5 MiB（≈18.5 MiB/行） |
| 指纹在真实权重上 | `model_fingerprint` 1 ms；`base_fingerprint` 0.46 ms（fp16 行只有 8 个数），两次加载一致，只改 25-49 块也能区分 |
| 同一基座的两个量化版本（int8 / w4a8） | `adaln_proj` 都是 `[96768,8] F16`、`adaln_t_table` 都是 `[1025,8] F32` → 同一个 `target_base`，一张表通吃 |

**云实例实测通过（2026-11-09，ComfyUI 0.44.0 / torch 2.9.1+cu130 /
RTX 4080 SUPER 32GB）**：

```
A 链 UNETLoader(minimax_h3_ref2va_int8_convrot, 34GB 全量教师)
   + DMAD 8-step LoRA + MiniMaxH3SigmaShift(12/2)
   + H3AdaLN(mode=record, for_model=ref2va_pruned_int8_convrot, fill_reference_row=True)
B 链 UNETLoader(minimax_h3_ref2va_pruned_int8_convrot, 20GB 剪枝基座)
   + 同 LoRA + 同 shift + H3AdaLN(mode=apply, table_in=A, after=A latent)
共享 CLIP / VAE / MiniMaxH3ImageToVideo(256x128, 5 帧) / BasicScheduler(8, simple)
```

- `H3LatentCompare`：**`VERDICT: PASS - bitwise identical`**，
  `samples__part0` / `samples__part1` 均 `equal=True max_abs=0.000e+00`；
- `apply stats`：`t rows=17`、`exact matches=408`（51 投影 × 8 步）、
  `match failures=0`、`native steps=0`；
- 录制日志：`51 projections hooked`、`extra rows: t=0.999 / t=1.000 pre-computed`、
  `for base: ... (fp ...) → saved h3_full_03692f_bf16.safetensors (17 rows, 314 MiB)`；
- 表头实读：`target_base` 已写、51 个投影、`out_dtype=torch.bfloat16`、
  t 集含 `0.999000013` 与 `1.000000000`；
- **第二个独立 prompt**（只有剪枝基座，`table=(auto)`、无 `table_in`）：
  report 显示 `apply · declared for this model`，`exact matches=408`、
  `match failures=0`、`native steps=0` —— 跨 prompt 自动查表成立；
- 离线四套测试在实例上全绿：`test_simple` / `test_merge` /
  `test_workflows` / `test_exact` + `test_runtime` 11/11。

**实测中发现并修掉的两个 bug**：

1. `fill_reference_row` 已把 `0.999` 行写进表，却仍打印「缺行」警告 ——
   `f32_image(0.999)=0.999000013` 与常量 `0.999` 差 1.3e-8，而检查用了
   1e-9 容差。改为按 `f32_image` 比较、容差 1e-6（commit `f314319`）。
2. 首次跨 prompt 查找失败：报「9 张表都没声明归属」。根因是 **AdaLN 权重
   文件为 F16，而 ComfyUI 以 bf16 计算**，fp16→bf16 丢尾数，导致
   `base_fingerprint_from_file`（录制侧）与 `base_fingerprint(dm)`（应用侧）
   不同。修复：基座指纹统一先舍入到 bf16 精度，两侧字节流一致；教师身份指纹
   不受影响（commit `22b61a1`），并加回归测试。重录后自动查找成功。

**v2 精简与边界加固（2026-11-09 收尾）**：

- 删掉 v1 七个单功能节点、`bake_engine.py`、`latent_io.py`、`tools/`
  全部生成器、01~08 工作流；`nodes.py` 由 1218 行瘦到 346 行（helper +
  `H3LatentCompare`）。只保留三节点 + 两份 API prompt。
- `table_dirs()` 不再扫 `output/h3_adaln/`：v1 遗留表会和新表竞争。
- **移除「唯一未声明归属的表」静默兜底**：v1 表可能属于别的教师，用了就是
  不可察觉的错答案。现在 `target_base` 必须命中唯一一张，否则明确报错并提示
  重录时指定 `for_model`。显式 `table` 覆盖仍可用。
- 读表失败包一层带路径的清晰异常；测试补了 fp16/bf16 指纹一致性、
  单目录扫描、无兜底行为等回归项。

**仍未验证**：`ON_CLEANUP` 之外的极端调度、以及带真实参考素材（关键帧/参考图/
参考音频）的完整出片 —— 目前 pin 行是在无参考素材的流程上预计算并在 tiny
模型上验证逐位一致，尚未在 34GB 教师上跑带参考素材的 A/B。

---

## 1. 项目目标与实现路径

**目标**：在固定调度下，让「剪枝/hybrid MiniMax-H3 + 精确 AdaLN 表 + LoRA」与
「完整 teacher + 同一 LoRA」的生成结果逐位一致（bitwise），速度维持剪枝版水平；
表可 Save/Load 分享，并适配 LoRA。

H3 有约 13B 参数（39%）在 AdaLN 分支里，其输出只依赖 timestep。剪枝版把这整条分支
折叠成 8 维 curve 基（`adaln_t_table [1025, 8]` + 每块 `[96768, 8]` 系数）。
8 维子空间装不下 2688 维的完整调制，所以必须**把每个 t 的最终调制值存成表**，
运行时替换掉 AdaLN 的输出。

两条产表路径：

| | `H3 AdaLN Record`（精确） | `H3 Exact AdaLN Bake`（近似） |
|---|---|---|
| 时机 | 真实采样过程中 | 离线，不采样 |
| 数值来源 | teacher 自己的 forward | 未 load 的 CPU 权重重算 |
| 相对原生 | **逐位一致（构造保证）** | 否（差 1 个 bf16 ULP） |
| 用途 | 任何要对比/发布的场景 | 便宜地探索 schedule |

**结论：Record 是唯一精确路径，Bake 只用于快速探索。**

---

## 2. v1 节点清单（已删除，历史参考）

当前只有三个节点，见 §0 与 README。下表是已删除的 v1 节点，仅用于读懂后面的排障记录：

| 节点 | 输入 → 输出 | 说明 |
|---|---|---|
| `H3AdaLNRecord` | full MODEL (+LoRA) → MODEL, TABLE | 采样期只读 hook，`time_embedder` 取 t，`adaln_proj` 取原生输出 |
| `H3ExactAdaLNBake` | full MODEL + sigmas → TABLE | 离线近似；支持 `splice` donor（plain float） |
| `H3ExactAdaLNSave` | TABLE → 文件路径 | 像 Save Image，自动编号；镜像到 `input/` |
| `H3ExactAdaLNLoad` | `table_name`（自由文本/子串/绝对路径）→ TABLE | 不用 COMBO（启动时缓存）也不支持上传控件 |
| `H3AdaLNMerge` | 两张 TABLE + `split` + `final_from` → TABLE | **bX-49 专用**：0..split-1 取 A，split..49 取 B |
| `H3ExactAdaLNApply` | pruned MODEL + TABLE → MODEL | `strict` / `fallback` / `off`；`off` 会先清理残留 hook |
| `H3LatentCompare` | 两个 LATENT → 报告 | **仍保留**；支持 NestedTensor，拒绝同源输入（防假 PASS） |
| `H3LatentSave` | LATENT → 文件 | 核心 `SaveLatent` 对 NestedTensor 会崩，必须用这个 |

---

## 3. v1 工作流清单（已删除）

01~08 及 `tools/build_*.py` 生成器全部删除。现在 `workflows/` 只有两份
**API 格式**的 prompt（提交到 `/prompt`，不是 UI 工作流导出）：

| 文件 | 用途 |
|---|---|
| `v2_record_apply_ab.json` | 单 prompt 双链：A 录表、B 注入、`H3LatentCompare` 逐位比较 |
| `v2_apply_auto.json` | 跨 prompt：只剪枝基座、`table=(auto)`、自动查表 |

两者都在云实例实测通过；`tests/test_workflows.py` 离线校验结构。



## 4. 已验证的事实（不要重复排查）

### 4.1 Record 路径已达成 bitwise

实例 ComfyUI 0.44.0 / RTX 4080 SUPER，同一个 API prompt 连跑三次 `05_record_strict.json`：

| 次序 | Apply `mode` | Compare | 注入日志 |
|---|---|---|---|
| 1 | `strict` | `PASS / max_abs=0`（90.83s） | `first injection confirmed` |
| 2 | `off` | `FAIL`（video `3.349e+00`、audio `2.970e-01`） | **无**（残留 hook 已清） |
| 3 | `strict`（off 之后） | `PASS / max_abs=0` | `first injection confirmed` |

报告落盘 `output/h3_adaln/compare_*_145323 / 145341 / 145405.txt`。

**关键**：第 2 次必须既 `FAIL` 又**没有**注入日志。若出现注入日志，说明对照组被上一次
的 hook 污染（该问题已修：`mode=off` 现在先清理共享 `nn.Module` 上的 runtime）。

### 4.2 已排除的差异来源

| 排除项 | 证据 |
|---|---|
| 跨次运行非确定性 | 同 prompt 双链仍 FAIL；开关 `--disable-comfy-compiler` 结果逐位相同 |
| Comfy 编译器 / graph breaks | 加 `--disable-comfy-compiler` 后 `max_abs=3.556 / mean=0.1449 / 31.14 dB` 完全不变 |
| 完整/剪枝模型的非 AdaLN 权重不同 | 829 个共同张量 shape/dtype 零差异；6 个 73 MiB 量化大权重 SHA256 全 SAME |
| 表上传 / Load / 资产面板 | Load 改自由文本；Save 支持 NestedTensor；核心 `SaveLatent` 已换掉 |
| 核心 `SaveLatent` 崩溃 | 已定位为 NestedTensor，插件自带 `H3LatentSave` |

### 4.3 根因（离线 Bake 为什么永远不精确）

实测：在实例上用 `comfy.sd.load_diffusion_model(FULL)` 加载完整模型，用表里的 17 个 t
重算 `blocks.0.adaln_proj.linear(F.silu(time_embedder(t).to(bf16)))`：

| 对比 | equal | max_abs | mean_abs |
|---|---|---|---|
| CPU 权重状态 vs 表 | False | 0.015625 | 6.66e-4 |
| CUDA 权重状态 vs 表 | False | 0.015625 | 5.24e-4 |
| CPU vs CUDA 互相 | False | 0.015625 | — |

`0.015625` = 一个 bf16 ULP。权重是 `QuantizedTensor`（`device=cpu`），CPU/CUDA 走
`comfy_kitchen` 不同 backend（eager 反量化 vs CUDA 量化 GEMM），数值不同。经 8 步扩散
放大成 latent 的 `mean_abs≈0.145`（PSNR≈31 dB）。**靠离线独立 Bake 永远复现不了采样时的
数值状态，所以改成记录式。**

源码层面的具体问题（历史记录，已修）：

| 位置 | 问题 |
|---|---|
| `bake_engine.py:130/134` | 在模型未经 `load_models_gpu`/`patch_model` 时直接调 `time_embedder`/`linear` |
| `comfy/model_patcher.py:921-922` | 无 LoRA 的量化权重根本不反量化 |
| `bake_engine.py:74` | `patch_weight_to_device(inplace_update=True)` 撞 `assert` |
| `bake_engine.py:116/140` | `clone()` 共享 `nn.Module`，`unpatch_model()` 污染原 ModelPatcher |
| `sigma_plan.py:78` | `round(float(t), 7)` 与 `model.py:743` 的未舍入 `unique_t` 不一致 |
| `apply_engine.py:164` | 表转 bf16 注入，但 curve 模型 `adaln_dtype=fp32` |

### 4.4 t 集合与 shift 的关系

严格相等的**充要条件**是「运行时所有 t ⊆ 表的 t 集合」+「t→t_emb 映射一致」。
`shift_v` / `shift_a` 相同既不充分也不必要：

- 不充分：shift 相同，但 Bake 用 8 步 simple、运行时用 10 步 ManualSigmas，t 集合不同照样失败。
- 不必要：把运行时的 `SIGMAS` 接进 Bake 的 `sigmas` 输入，或用 `extra_t` 补齐实际 t，就能全命中。
- ManualSigmas 绕开了 shift 对 sigma 网格的影响，但 `t_a = 1-time_shift_sigma(σ, shift_v, shift_a)`
  和 `audio_scale = shift / audio_shift` 仍然读 shift。
- `_auto_shifts` 解析顺序：`transformer_options` → `dm.sigma_shift_*` → `model_sampling`，
  与 `model.py:672-673` 对齐。

---

## 5. bX-49 混合模型（b25-49）完整流程

### 5.1 文件结构事实（实测）

`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors`（19.53 GiB，932 张量）：

- **没有 `time_embedder`**
- 51 个 `adaln_proj.linear.weight` 全是 `[96768, 8]`（F16），`adaln_t_table [1025, 8]`
- 配方（见 `_build_b25-49_w6a8.py`）：**fl2va_pruned 骨架 + ref2va_pruned 的 blocks 25-49
  那 50 个 adaln 张量**，不是「full 块 + pruned 块」拼接

**所以 b25-49 整份文件都是 curve 形态，Record 跑不了，8 维也装不下完整调制。**
不能靠「特判文件」自动还原；必须引入第二个 teacher。

### 5.2 正确做法：两个 full teacher 各录一张表，再按块合并

1. **FL2VA full**（有 `time_embedder`）+ 同一 LoRA，同一调度 → A 表
   （`08_record_fl2va.json`，约 75s）
2. **Ref2VA full** + 同一 LoRA、同一调度 → B 表
   （`06_b25-49_hybrid.json` 的 B 链，约 67s）
3. `H3AdaLNMerge`，`split=25`、`final_from=a`：
   blocks 0-24 ← A，25-49 ← B，`final_layer` ← A
4. 合并表 Apply 到 b25-49

表只依赖 t，所以两次 Record 不必同一次运行、不必同 seed，只要 t 集合一致。
每块仍是对应 teacher 的原生输出，逐块精确（不是 8 维近似）。

**为什么不能直接拿 FL2VA 全表套上去**：会把 25-49 块也变成 FL2VA 调制，抹掉 ref2va 的
参考条件能力。数值上「更接近 full」，语义上是错的。对照实测（同一 b25-49）：

| 注入 | video | audio |
|---|---|---|
| hybrid 表 | 基线 | 基线 |
| FL2VA 全表 | `max_abs=4.385 / mean=0.470 / 22.42 dB` | `0.935 / 0.107 / 27.72 dB` |

差异显著，说明尾部 25 块的 ref2va 调制确实生效。

### 5.3 实例验证结果（2026-11-08）

- FL2VA 表 75.3s → `fl2va_dmad8_00001.safetensors`（15 行 / 51 投影 / 277.5 MiB，
  fp `001c095c7e10a160`）
- Ref2VA 表 67.4s → `ref2va_dmad8_00001.safetensors`
- Merge 3.8s → `b25-49_hybrid_00001.safetensors`
- **表级校验**：0-24 == FL2VA、25-49 == Ref2VA、`final_layer` == FL2VA，
  15 行全部 `torch.equal`
- `b25-49 + Apply(hybrid)` 55.1s：`hooked modules 51`、`lora fp match`、
  `first injection confirmed ... max_dist=0.000e+00`，latent 正常落盘

### 5.4 注意事项

- **必须有 Ref2VA full（34GB，带 `time_embedder`）**；只有 pruned 版无法还原。
- 两表必须同 t 集合 / 同 LoRA / 同 strength / 同 `out_dtype`；Merge 会拒绝并列出缺失 t。
- b25-49 是反量化后注入，目标量化档位不影响注入正确性；但 `out_dtype` 必须与 teacher 一致。
- ~~端到端 bitwise 对照暂无~~ **已补（2026-11-08 晚，见 §11）**：已构建并发布全量 b25-49
  hybrid teacher，端到端跑出 `max_abs=0.000e+00 / psnr=inf`。

---

## 6. 代码结构（当前）

```
h3_unified.py      H3 AdaLN (record/apply/off) + H3 AdaLN Hybrid
nodes.py           共享 helper + H3 Latent Compare（346 行）
record_engine.py   采样期只读 hook（精确路径）
apply_engine.py    运行时注入：t_emb 匹配 + sigma 兜底
sigma_plan.py      float32 精确的 t 运算，镜像 model.py:671-751
table_io.py        safetensors 容器 + merge_tables（bX-49 合并）
workflows/         两份 API prompt（v2_record_apply_ab / v2_apply_auto）
tests/             test_simple / test_exact / test_merge / test_workflows /
                   test_runtime
```

v1 的 `bake_engine.py`、`latent_io.py`、`tools/`、01~08 工作流已删除；
下面的实现要点中若提到它们，属于历史根因记录。

几个容易踩的实现点：

- **Apply 是 hook `AdalnProj` 的 forward**，直接返回 `chunk(expand)` 后的 6 个调制表，
  所以表与目标模型是 full 还是 curve 无关。
- **匹配是精确的**：key 是 t 的 float32 像（`struct.pack`），容差 `1e-6 × |keys|`，
  远小于 1 个 bf16 ULP；不精确就报错，不静默吸附到相邻行。
- **hook 按共享 `nn.Module` 跟踪**（`_h3_recorders` / `_h3_exact_runtimes`），
  因为 `ModelPatcher.clone()` 不复制自定义属性但共享 `nn.Module`；不这样做会跨 prompt 累积。
- **`mode=off` 先清理旧 runtime 再早退**，否则对照组会被上一次的 hook 污染。
- **LoRA 指纹**：新表存权重摘要 + strength（`lora_fingerprint`）和 keys-only
  （`lora_fingerprint_keys`）。比较顺序：先比 fp，再对旧表回退 keys。中间版本表
  （有 fp、无 keys）靠 fp 相同通过。
- 表 fp32 存储，回放时按记录的 `out_dtype` 转回（bf16 存储会引入 ~2.5e-2 相对误差）。

---

## 7. 模型与文件清单（换机后要备齐）

| 角色 | 文件名 | 说明 |
|---|---|---|
| FL2VA full teacher | `minimax_h3_fl2va_int8_convrot.safetensors` | 有 `time_embedder`，b25-49 的 0-24 来源 |
| Ref2VA full teacher | `minimax_h3_ref2va_int8_convrot.safetensors` | 有 `time_embedder`，25-49 来源；05 用的就是它 |
| pruned target | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 05/02 的 B 链 |
| hybrid target | `minimax_h3_hybrid_fl2va_ref2va_b25-49-int8.safetensors` | 20GB，全 curve，无 time_embedder |
| LoRA | `dmad_minimax_h3_8step_full_critic_comfyui.safetensors` | 8 步、shift 12/2、strength 1.0 |
| 文本编码器 | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | |

已产出的表（实例 `output/h3_adaln/`）：

- `fl2va_dmad8_00001.safetensors`（A 表，FL2VA full）
- `ref2va_dmad8_00001.safetensors`（B 表，Ref2VA full）
- `b25-49_hybrid_00001.safetensors`（合并表，可直接分享）
- `recorded_00001/00002.safetensors`（历史：05 在 Ref2VA full 上录的）

---

## 8. 关键坑（换机必看）

1. **不要提交 SSH 凭据。** 仓库是公开的；主机的端口和密码每次开机都可能变，由用户提供。
2. **GitHub 访问**：本机 IPv6 黑洞，`curl` 需 `-4`；schannel 吊销检查失败时加
   `--ssl-no-revoke`。`git push` 走 HTTPS 可用（有 credential helper）。
3. **`pkill` 会杀掉承载它的 SSH 会话**：`pkill -f "[m]ain.py --listen"` 会匹配到
   `bash -c` 命令行里的同一串字符。用 `pkill -f "^python main.py"`，并且**kill 与
   nohup 启动分成两条连接**，不要在同一条命令里既 kill 又 start。
4. **`-py` 传远程绝对路径要 `MSYS_NO_PATHCONV=1`**，否则 Git Bash 会把 `/root/...`
   改写成 Windows 路径；本地脚本路径写成 `D:/...` 形式。
5. **重启 ComfyUI 会清空 `/history`**。要复现某个 prompt，先把 `/history` 里的
   API prompt 存成文件（`/root/h3_05_prompt.json` 就是这么来的）。
6. **05 的 teacher 是 Ref2VA full**，不是 FL2VA。06 的 A 表必须是 FL2VA（用 08 录），
   别把两者搞混，否则合并出「全 Ref2VA」的表。
7. **核心 `SaveLatent` 对 NestedTensor 会崩**（`nodes.py:558` `.contiguous()`），
   用插件的 `H3LatentSave`。
8. **`H3ExactAdaLNLoad` 用自由文本**，不要用 COMBO + `image_upload`（前者启动时缓存，
   后者只收图片）。
9. **表与 t 集合绑定，不与分辨率绑定**：换步数 / 调度器 / shift / AdaLN-LoRA 强度会改变需要的 t（缺了就报 t-set mismatch，再录一次是**追加行**）；分辨率、帧数、seed 不改变 t，不需要重录。
10. **本地 ComfyUI 0.41.2 跑不了 `test_exact.py`**（低于 0.42.5，且缺 `comfy_kitchen`）；
    数值回归要在 0.44 实例上跑。本地能跑的是 `test_merge.py`（纯 CPU）和
    `test_workflows.py` 的结构部分。

---

## 9. 复现与验证命令

```bash
# 结构：所有工作流 JSON、连线/槽位、widget 数量、命名参数一致性
python tests/test_workflows.py

# 数值：真实 MiniMaxH3Model 微型配置，teacher vs curve student（需要 ComfyUI）
COMFYUI_DIR=/root/ComfyUI python tests/test_exact.py

# 合并：分块、t 对齐、元数据往返、各种拒绝路径（纯 CPU）
python tests/test_merge.py
```

实例上更新与重启：

```bash
cd /root/ComfyUI/custom_nodes/ComfyUI-H3-ExactAdaLN
git -c gc.auto=0 -c maintenance.auto=false pull --ff-only origin main
cp workflows/*.json /root/ComfyUI/user/default/workflows/

pkill -f "^python main.py"      # 单独一条连接
# 然后另开一条：
cd /root/ComfyUI && (nohup python main.py --listen 0.0.0.0 > comfyUI.log 2>&1 < /dev/null &)
```

---

## 10. 已知限制与未做项

- 表是 (base 权重, LoRA 集合与强度, sigma 网格, shifts, packed layout) 的函数；
  其中任何一项变化都要重录。
- `out_dtype` 在产表时固定，必须与 teacher 推理 dtype 一致。
- per-row `denoise_mask` 的 timestep 不被 Record 支持（Bake 可用 `extra_t` 覆盖）。
- HyperFlow 式 `(t, r)` 端点条件不复现。
- Python hook 到不了 C++ 引擎或绕过 `module.forward` 的融合 kernel。
- 低优先未修：跨次 schedule 预检、重复 t 只校验 canary `blocks.0`、`_cur_t` 非线程局部、
  bake 的 splice 分支绕过 LoRA、`dit_patch` 通道未启用。

---

## 11. 全量 b25-49 teacher、time_embedder 坑、LoRA 兼容性（2026-11-08 晚）

### 11.1 已发布的全量混合模型

`binglingzhimeng/MiniMax-H3-b25-49-full`（HF，34,038,892,334 B，
`sha256=35a57db8…8f84f`）：FL2VA full 整机 + Ref2VA full 的 blocks 25-49
`adaln_proj`（4 张量/块，共 100 个）。做法是「复制基座 + 原地打补丁」，尺寸不变，
所以 `data_offsets` 与基座完全一致。校验：100/100 替换张量 == Ref 源字节，
其余 25.63 GiB 逐字节 == FL 源，HF 上 LFS oid 与本地 sha256 一致。

**端到端验证（同 LoRA、8 步 simple、shift 12/2、832x480x39）**：

```
A: full b25-49 hybrid (native)
B: pruned b25-49 + table recorded on A
samples__part0 / part1: equal=True max_abs=0.000e+00 psnr=inf
VERDICT: PASS - bitwise identical
```

### 11.2 关键坑：两个 teacher 的 time_embedder 并不相同

`fl2va_int8_convrot` 与 `ref2va_int8_convrot` 的 `time_embedder.proj_out.weight`
最大差 **0.104**（`proj_in` 0.071）。所以老配方「FL 表 + Ref 表合并」在尾部
（blocks 25-49）拿到的是 `W_ref @ e_ref(t)`，而真正的 b25-49 吃的是
`W_ref @ e_FL(t)`。实测差异：

| 尾部来源 | blocks 25-49 与全量模型的最大偏移 |
|---|---|
| Ref 原生录制（老配方） | 0.09 – 0.19 |
| Ref + FL 的 time_embedder（新配方） | **0**（25/25 逐位） |

端到端：老合并表注入 pruned vs 全量模型 = `22.20 dB` / `31.10 dB`。

### 11.3 新配方：`H3AdaLNRecord.time_embedder_from`

不用 34 GB 合并文件也能得到目标 A：

1. `08` 录 FL2VA full → A 表；
2. 录 Ref2VA full 时把 `time_embedder_from` 填成 FL 的 checkpoint → B' 表；
3. `H3AdaLNMerge`（`split=25`、`final_from=a`）。

实测：B' 的 blocks 25-49 与全量模型录出的表 **25/25 逐位一致**，
模拟合并 `A + B'` 与全量表 **51/51 全等**。

实现要点（两个都必须，缺一就会静默失败）：

- **不能用 weight patch**。`patch_weight_to_device` 会把权重降到
  `lora_compute_dtype`（bf16），而真模型里 embedding 是 fp32，单这一项就带来 ~1e-1 偏移。
  必须 `add_object_patch("diffusion_model.time_embedder", deepcopy(...))` 整体换模块。
- **hook 要挂到换上去的那个模块**。object patch 之后 `dm.time_embedder` 仍是旧对象，
  否则采样时报 `blocks.0 fired before time_embedder`。`AdaLNRecorder` 因此新增
  `te_override` 参数。

护栏（2026-11-08 晚补，均实测过）：

| 情况 | 行为 |
|---|---|
| 供体张量名字/形状对不上 | 硬报错（**部分替换会混两个嵌入**，绝不允许） |
| 供体没有 `time_embedder.*`（pruned 文件） | 硬报错，提示不能当嵌入源 |
| 模型上已有 `time_embedder` 的 LoRA patch | 硬报错（模块替换会把这部分 delta 丢掉） |
| 同一个 MODEL 上挂两个 Record | 只保留最后一个（前者表为空），日志告警 |
| 先带 `time_embedder_from` 录过、再不带地录 | 先撤销上次的模块替换，第二次录到的是供体原生值（实测 50/50 对得上 `t3_ref`） |
| 模块替换不可用，退回 weight patch | 报告里写明 `weight patch - bf16, NOT bitwise`，不再假装是精确路径 |

合并表新增 `hybrid_te_from_a` / `hybrid_te_from_b` 两个元数据；如果 split>0
而 B 没带嵌入覆盖，Merge 的报告会明确标出「这是老配方」。

### 11.4 pruned b25-49 的真实构成（逐张量核对）

`minimax_h3_hybrid_fl2va_ref2va_b25-49-int8`：

- blocks 0-24 `adaln_proj` == `fl2va_pruned`（逐位）
- blocks 25-49 == `ref2va_pruned`（逐位）
- `adaln_t_table` == `fl2va_pruned`（逐位），当前无 `time_embedder`

即「基座身体 + 供体尾部旋钮」，这也说明目标 A 才是它的忠实数版本。
注意：它自己的曲线输出反而更接近老合并表（18/18 采样行更近，mean abs
0.003 vs 0.008）—— 因为拿基座的坐标表去喂供体的 8 维尾部权重本身就是近似。

### 11.5 LoRA 兼容性（扫了实例上 104 个唯一 H3 LoRA）

| 类别 | 数量 | 输入宽 | 结论 |
|---|---|---|---|
| 无 adaln / time_embedder 键 | 87（84%） | — | 一张表通吃 |
| 只碰 `adaln_proj` | 7（7%） | 8（curve 形态） | 能装 pruned，会真改调制 → 要单独录 |
| 只碰 `adaln_proj` | 10（10%） | 2688（full 形态） | 装 pruned 时被丢弃 → 想保留其 AdaLN 效果要在教师侧录 |
| 只碰 `time_embedder` | 0 | — | — |

规律：**加速/蒸馏类**（turbo、4 步、8 步 Acc、dmd）全都带 AdaLN 键；
风格/动作/特效/滑块类全都不带。DMAD 8 步 LoRA 属干净的一类。

跨 LoRA 实测：用 DMAD 8 步录的表，配 turbo 8 步跑 pruned，对照全量 + turbo 原生
`equal=True / psnr=inf`。

### 11.6 Apply 的 strict 判定已放宽

非 AdaLN 的 LoRA 指纹不再让 `strict` 硬失败（只报告「differs - harmless」），因为
调制只由 `adaln_proj` / `time_embedder` 决定。仍然硬失败：AdaLN/时间步指纹不匹配、
显式 `expected_teacher` 不匹配、t 集合不匹配。此前跨 LoRA 场景只能用 `mode=fallback`。

### 11.6b Apply 的 dtype 与教师匹配提示（2026-11-08 晚）

- Apply 现在把 `table.out_dtype` 和目标模型的计算 dtype 对比（只取
  `manual_cast_dtype` 与 `model.get_dtype()` 两个可信来源，不用参数 dtype 猜，
  避免 norm 的 fp32 造成假报警），不一致时报告里写
  `MISMATCH: model computes in bf16 -- replay is NOT bitwise` 并打日志。
  实测：把表元数据改成 `torch.float16` 后告警正常触发。
- `expected_teacher` 仍是子串匹配，但当匹配不是完整字符串时，报告里会标
  `(substring match)`，方便确认自己匹配到了哪张表。
- `time_embedder_from` 传绝对路径且文件不存在时，报错改成
  「does not exist (absolute paths are used as-is, with no search)」，
  不再误导成「在 diffusion_models 里没找到」。

### 11.6c 统一节点 H3AdaLN（2026-11-09，实验性，已注册）

> **已被 §0.1 的 v2 取代**：五种模式合并成 `apply / record / off`，
> 四个 tag、`table_source`/`table_text`、`strictness` 全部删除，
> `hybrid`/`merge` 独立成 `H3 AdaLN Hybrid`。下面这段保留为 v1 的设计记录。

一个节点五种模式：`apply / record / hybrid / merge / off`。表统一放
`models/h3_exact_adaln/`，`record` 在 prompt 结束时（`ON_CLEANUP`）自动落盘，
不需要 Save 节点，也不需要 `after` 连线。

- **命名**：`{model}__{lora}__[pass__]{scheduler}{steps}__sh{sv}-{sa}__{fp6}.safetensors`，
  不含 t 行数（同配置重录覆盖），行数进元数据。
- **记录→推理无缝**：存盘后写 `_last_table.json`；`table_source=auto` 时下拉项
  失效则自动退回最近录制，切个模式就能用。
- **二采/高低模**：不做自动判定，两个节点各挂各的模型 + 各自的 `pass_tag`；
  另有 `model_fingerprint` 兜底，拿错表直接报错。
- 已知未验：`ON_CLEANUP` 时机、COMBO 是否刷新、二采与 SelfLift 的实际行为。
  清单见 `docs/unified-node.md` §8 / §8.1。

### 11.6d YZ 终极篇工作流 / SelfLift 实测（2026-11-09）

把 `H3AdaLN` 插进 `YZ-MiniMax H3 终极篇-文武双全.json`（103 节点、
Get/Set 虚拟总线、SelfLiftH3Sampler），插在 `模型` 总线之前
（ModelPreviewOverrideKJ → H3AdaLN → SetNode #39）。
由于参考媒体文件不在实例上，而 `MiniMaxH3ReferenceToVideo` 的 ref_* 全是
optional，模拟时裁成 18 节点的采样主干（256x144 / 5 帧 / 4 步），两条链共用
conditioning / sigmas / seed。

**跑通的**：
- `record` 在 `ON_CLEANUP` 落盘（3 个 t 行、55 MiB、0 冲突、2 次 forward），
  `_last_table.json` 写对；切成 `apply` + `table_source=last` 零改动直接读到。
- **SelfLift 两段连续采样被一张表完整覆盖**：102 次注入
  （51 投影 × 2 次 forward），索引 `[0]` 与 `[1,2]`，`exact matches=102`、
  `match failures=0`、`native steps=0`、`max_dist=0`。这回答了 §2.3。
- 指纹检查修掉一个真 bug：full 教师与 pruned 基座的 adaln_proj 形状不同，
  相等检查会误杀主路径；现在只对 full 目标比较。

**没达到的**：full 原生 vs pruned+表 **不是逐位一致**。
对照测量（同 prompt 双链、同 seed）：

| 组合 | video PSNR | audio PSNR |
|---|---|---|
| 全量原生 vs 剪枝原生（都不加表） | 27.24 dB | 25.70 dB |
| 全量原生 vs 剪枝 + 表 | 36.05 dB | 29.10 dB |

表把差距拉近 6~9 dB，但仍有 ~30 dB 残差。注入本身是完美的
（102/102 精确命中、0 回退、0 native 步），所以残差不在表的覆盖范围。
当时推测是模型对/精度问题 —— **§11.6e 证明这个推测不对**：同一实例、
同一对模型、换成没有 SelfLift 的 DMAD 工作流后直接逐位 PASS。
残差来自 SelfLift 自身的高/低分辨率机制，不是模型对。

### 11.6e DMAD8 步工作流（无 SelfLift）对照 → 逐位 PASS（2026-11-09）

`DMAD8步-文戏专用.json`：35 节点、标准 `SamplerCustomAdvanced +
BasicGuider`，5 个 LoRA（含 dmad8 @1.0）、simple 8 步、lcm、
shift 12/2。同样裁成低开销采样主干（256x144 / 5 帧）并保留完整模型链
（SageAttentionPatch → 5 LoRA → SigmaShift → ModelAttentionBackend），
两条链共用 conditioning / noise / sigmas / seed。

- 重新在全量 teacher 上录表：15 个 t 行、0 冲突、8 次 forward、bf16。
- `record` → 切 `apply` + `table_source=last`，零改动衔接。
- **A 链全量原生 vs B 链剪枝 + 表**：
  `samples__part0 equal=True max_abs=0.000e+00 psnr=inf`、
  `samples__part1 equal=True max_abs=0.000e+00 psnr=inf`，
  **VERDICT: PASS - bitwise identical**。
- 负对照（B 链 `mode=off`，即不加表）：30.48 dB / 29.87 dB，FAIL。
  说明 PASS 确实是表带来的，不是两条链碰巧相同。

**结论**：表与节点本身在标准采样路径上完全正确；§11.6d 的残差是
SelfLift 特有的。已把 `H3AdaLN` 插进两份 UI 工作流（YZ 版插在
`模型` 总线前，DMAD 版插在 `ModelAttentionBackend` 之后），
成品在 `D:/下载/*_H3AdaLN.json`。

### 11.6f 导演台二采（不同模型 + 自定义 sigmas）→ 分表机制验证通过（2026-11-09）

`导演台二采+自定义sigmas (2).json`：20 节点，一采 `b25-49-int8`、
二采 `ref2va_pruned_int8_convrot`（**两个不同模型**），各自 `ManualSigmas`
（一采 `1,0.86,0.55,0`、二采 `0.35,0.2084,0`），提示词内嵌在
`H3SeamlessChainSampler` 的「导演台状态」JSON 里，没有 CLIPTextEncode 节点。
（用户先给的 (1) 版二采是 `ref2va_pruned_w4a8_mixed`，没有对应全量教师，
因此换成 (2) 版验证。）

**两个节点各挂各的模型链，各自 `pass_tag` 造表**：

| 表 | 教师 | t 行数 | model fp | lora fp | pass_tag | t 集合 |
|---|---|---|---|---|---|---|
| `seq_p1_lowres` | b25-49 full | 5 | `e8008a9690aa` | `86c8f90a…` | p1 | 0, .14, .45, .494, .831 |
| `seq_p2_lowres` | ref2va full | 4 | `03845c348585` | `001c095c…` | p2 | .65, .792, .881, .938 |

两表指纹不同、t 集合不重叠、0 冲突 —— 证明 hook 不会串台，
**一采/二采确实各采各的**。两个 `record` 节点在同一 prompt 里都触发了
`ON_CLEANUP` 并各自落盘。注意 `_last_table.json` 会被后写的覆盖，
所以多 pass 应用时必须用 `table_text` 显式指定，不能靠 `auto/last`。

**结果（同 prompt 双链，A 链两个全量教师原生，B 链两个剪枝基座 + 各自表）**：

| 组合 | video | audio | 注入 |
|---|---|---|---|
| 有表 | `equal=True psnr=inf` | `equal=True psnr=inf` | 2 次 |
| 无表（mode=off） | 14.37 dB | 16.25 dB | 0 次 |
| 交叉表（一采用 p2、二采用 p1） | strict 报错 `t-set mismatch`（一采 t=0 不在 p2 表） | — | — |

说明分表不仅「能工作」，而且**必要**：两个 pass 的 t 集合不重叠，
互换会被 t-set 校验挡住。这也进一步收窄了 §11.6d 的结论 ——
普通的「一采 + 二采 + 不同模型」不是问题，残差确实是 SelfLift 特有机制。

**两个验证路上的坑（重要）**：

1. `H3SeamlessChainSampler` 的 `图像` 输出在「自动保存=分段」模式下是
   **全零占位张量**，真实 latent 写到
   `output/h3_projects/<存档目录>/seg_000.pt`。拿 IMAGE 输出比较会得到
   假的 `equal=True`。
2. 导演台按「存档目录」缓存分段，且有全局 `h3chain_state.json`；
   换模型/换表不会让它重算。跨 prompt 比较会读到旧结果（同样是假 PASS）。
   **必须每次用全新目录名 + 删 `h3chain_state.json` + 重启 ComfyUI 清内存缓存**，
   并比较 `seg_000.pt` 里的 `video` / `audio` 两个张量。
3. `BlockSparseAttention` 是动态 COMBO（widget 值与 API 输入不一一对应），
   构造 API prompt 时两条链同等跳过；它只是注意力性能优化，不影响 A/B 判定。

成品工作流：`D:/下载/导演台二采+自定义sigmas (2)_H3AdaLN.json`
（22 节点，一采/二采各插一个 `H3AdaLN`，已填好 `table_text` 与 `pass_tag`）。

### 11.6g SelfLift 源码排查：残差与插件无关（2026-11-09）

读 `comfyui-SelfLift` 源码定位 §11.6d 的 ~30 dB 残差。

**SelfLift 在两段采样之间插入了一个转换点**（`nodes.py:229`
`progressive_sample`），流程是：

1. 低分辨率段 `comfy.samplers.sample(low_model, ..., sigmas[:k+1])`；
2. 取最后一次评估的 x0 作为 clean endpoint（Eq.3，最后一次 Euler 更新被丢弃）；
3. `paired_lifts()`：要么走 VAE decode → 放大 → encode，要么走**外部
   学习式上采样器** `LatentResizer3D`（`h3_upscaler.py`，12 个 3D ResBlock，
   `eval()` 关掉 dropout）。YZ 工作流是 `rho=0` +
   `h3_upscaler_lms_v0.1.safetensors`，因此走后者、跳过 VAE 往返；
4. `artifact_aware_consistency_lift()`（rho=0 时直接返回 z_lat）；
5. `prepare_noise(z0_high, seed+1)` 重加噪 + `noise_scaling` + `_euler_step`
   重建状态（`comfy/sample.py:27` 用局部 `torch.manual_seed`，确定性）；
6. 高分辨率段 `comfy.samplers.sample(high_model, resume_noise=全零, ...,
   sigmas[k:])`。

`low_model`/`high_model` 只在有 noise_mask 时才 `clone()`；YZ 没有 mask，
所以两条链都直接挂在传入的 MODEL 上，hook 不会丢。`model_hires=None` 时
高段沿用同一个模型 —— 这也是为什么一张 3 行表能覆盖两段。

**定位实验**：临时把 `_debug_dump` 改成保存原始张量（已恢复源码），
在干净环境重新录表并跑「A 链全量原生 / B 链剪枝+表」同 prompt 双链：

| 转换点张量 | shape | A vs B |
|---|---|---|
| `z0_low`（低段 clean endpoint） | (1,24,2,4,8) | **逐位一致** |
| `z_lat`（学习式上采样器输出） | (1,24,2,9,16) | **逐位一致** |
| `z0_high`（校正后终点） | (1,24,2,9,16) | **逐位一致** |

即低段模型输出、独立上采样器网络、校正结果全一致；分叉只发生在
**转换点之后**（重加噪 + 高分辨率段）。而表在高段也是 102/102 精确注入、
0 回退、0 native 步。同模型双链在干净环境下逐位 PASS，排除了双链非确定。

**初步结论**：残差不在表的覆盖范围；判断表是否生效应看注入日志
（`exact matches` / `native steps`），而不是最终 latent 的 PSNR。
注意：这里说的「双模型」指为了对照而构造的 A 链（全量教师）/ B 链（剪枝基座），
用户工作流里 SelfLift 高低段本身是同一个模型（`model_hires` 为空）。
§11.6h 用单链独立运行重做了定位，结论见下。

**做了一处适配**：`H3AdaLN` 加回 `after` 排序输入（仅用于拓扑排序，不参与
计算）。SelfLift / 导演台这类采样器没有自己的 `after` 接口，用它可以把
「同一个 prompt 内录表 → 紧接着用表」的两条链强制排序，让 `table_in`
直连可靠工作，避免跨 prompt 时显存状态不同带来的额外扰动。

### 11.6h 单链独立运行复核 + 插件侧适配尝试（2026-11-09）

用户指出：SelfLift 高低段本来就用同一个模型，为什么还不能适配？
因此重新用**两次独立的单链运行**复核（A：全量教师 + record，B：剪枝 + apply，
各自只有一个模型在场，排除同 prompt 双链显存干扰），并临时把 SelfLift
的 `_debug_dump` 扩展到转换点之后（源码已恢复）。

| 阶段 | A vs B |
|---|---|
| 转换点 `z0_low` / `z_lat` / `z0_high` | 逐位一致 |
| 重加噪后的高段输入 `resume_latent`（video + audio） | **逐位一致** |
| 高段 `comfy.samplers.sample` 输出 `out` | FAIL 17.76 dB / 17.37 dB |

即**高段模型 forward 的输入完全相同，输出不同**。逐项排除：

| 检查 | 结果 |
|---|---|
| 非 AdaLN backbone 权重（attn/mlp，400 项抽样） | 400/400 逐位一致 |
| `comfy_quant` 反量化描述符（200 项） | 200/200 一致 |
| `weight_scale`（200 项） | 200/200 一致 |
| 表注入 | 102/102 精确命中、0 回退、0 native 步 |
| 表精度 | 内部 fp32 保存，按全量教师原生 bf16 注入；**试改为按目标原生 fp32 注入反而更差**（10.21 dB vs 17.76 dB） |
| 关闭 Apply 的 sigma wrapper | 结果完全不变（17.76 / 17.37 dB） |
| 同模型双链（干净环境） | 逐位 PASS，排除双链非确定 |

两个 checkpoint 唯一不同的就是 AdaLN 分支本身（`adaln_proj.linear.bias`
形状/dtype 不同，正是表要替换的部分）。也就是说：输入、权重、量化方案、
AdaLN 注入全部一致，输出仍不同 —— 剩下只能归因于 **ComfyUI 对 34GB 全量模型
（动态权重加载 / offload）与 20GB 剪枝模型（可全程驻留）采用了不同的执行路径**，
在 SelfLift 的 continuation sampling（从中间 sigma 继续、输入是半去噪 latent）
这个敏感状态下产生 1 ulp 级差异并被放大。这属于执行/显存层，**不是插件能适配的**；
试过的两种插件侧改法（dtype 跟随、关 wrapper）都无效或更差，说明当前实现已是最优。

**对实际使用的影响**：用户出片时只跑一条剪枝链，表正常工作；失真的是
「全量 vs 剪枝双链逐位一致」这个验收指标 —— 参照链本身受 offload 影响。
要真正消除，只能让两个 checkpoint 走同一显存路径（不现实：34GB > 32GB 显存），
或改用同源的、同样可全驻留的参照模型。保留 §11.6g 的 `after` 适配（同 prompt
录用的排序控制），但不为这个残差改注入逻辑。

### 11.7 尚未做

- ~~b15-49 / b20-49 等其他切分的全量 teacher~~ 已出 b5/b20/b25/b30/b45，发布在 `binglingzhimeng/MiniMax-H3-b25-49-full`。
- 只在 ComfyUI 0.42.5 / torch 2.9.1 上验过；`manual_cast_dtype` 打开时
  object patch 是否会绕过 cast 未测，遇到异常优先怀疑这条。
- ~~`tools/build_hybrid.py` 还没把 `time_embedder_from` 写进生成的 06~~ 已补。
- 模型卡里已写明「与老合并表不是同一目标」及全部实测数字。

---

## 12. 经验教训清单（2026-11-08/09，换机必读）

完整教程见 `docs/NODE_GUIDE.md`。这里只列最容易再踩的坑。

### 12.1 表与模型

- 表是 `t -> modulation` 的纯函数，**按模型分表，不按 pass 分表**。
  同一模型的多段采样共用一张表（t 并集）。
- 不同模型才用两个节点 + `pass_tag`；两段 t 集合不重叠时表不能互换
  （交叉表会被 strict 的 t-set 校验挡住）。
- 87/104 的 H3 LoRA 不碰 AdaLN，一张表通吃；加速蒸馏类才需重录。
  非 AdaLN 指纹不匹配**不**该让 strict 失败（已修）。
- 录 donor 教师必须 `time_embedder_from`，且必须整体换模块；weight patch
  会降到 bf16 造成 ~1e-1 偏差。hook 要挂替换后的模块。
- 全量教师与剪枝基座的 adaln_proj 形状不同，指纹相等检查会误杀主路径（已修）。
- 表内部 fp32 存、按教师原生 bf16 注入；改成按目标 fp32 注入反而更差
  （10.21 dB vs 17.76 dB）。**别再改 dtype**。
- 每行 ~19.7 MiB；多尺度流程行数多时注意磁盘。

### 12.2 验证方法（血的教训）

- **负对照必需**：`mode=off` 必须 FAIL，否则 PASS 不可信。
- **ComfyUI 缓存会造假 PASS**：看 `first injection confirmed` 计数，
  为 0 就是没重跑。换 seed / 重启 / 改无关 widget。
- **判据是注入日志**（`exact matches` / `native steps`），不是最终 latent PSNR。
- 导演台 `图像` 输出是全零占位，真实 latent 在
  `output/h3_projects/<存档目录>/seg_000.pt`；按目录缓存，换模型不重算。
  比对前要全新目录 + 删 `h3chain_state.json` + 重启。
- UI 工作流 ≠ API prompt：Get/Set 是前端虚拟总线；动态 COMBO 节点
  （`BlockSparseAttention`）widget 与 API 输入不对应。
  `MiniMaxH3ReferenceToVideo` 的 `ref_*` 全是 optional，可裁掉缺失素材。
- 对照链必须共享 conditioning / noise / sigmas / seed。
- 改第三方源码做诊断要备份并恢复，改动后重启 ComfyUI。

### 12.3 边界

- SelfLift 残差：单链独立运行逐阶段 dump 证明转换点前逐位一致，
  分叉只在高段 forward；backbone / comfy_quant / weight_scale 全部一致，
  表 102/102 精确注入。归因 34GB full（offload）vs 20GB pruned（全驻留）
  的执行路径差异，**不是插件能适配的**；试过 dtype 跟随、关 sigma wrapper
  都无效或更差。**不要再往这个方向改注入逻辑。**
- 失真的是「全量 vs 剪枝双链逐位一致」这个验收指标；实际出片只跑剪枝链，
  表正常工作。
- `_last_table.json` 多 pass 录制会被覆盖 → 应用侧用 `table_text` 显式指定。
- `after` 输入只排序不参与计算，给无 after 接口的采样器做同 prompt 录用。

### 12.4 今天修掉的真 bug

| Bug | 修法 |
|---|---|
| 指纹相等检查误杀 full→pruned 主路径 | 只对 full 目标比较 |
| 部分 `time_embedder` 替换会混两个嵌入 | 全量匹配校验，不匹配硬报错 |
| 上次 object patch 残留到下次录制 | 进节点先撤销 |
| 降级 weight patch 却报「PATCHED」 | report 明写 `NOT bitwise` |
| 多 Record 节点静默顶掉前一个 | 日志告警 |
| 合并表丢失嵌入来源元数据 | 新增 `hybrid_te_from_a/b` |

### 12.5 发布物

- GitHub：`bingling360/ComfyUI-H3-ExactAdaLN`，教程 `docs/NODE_GUIDE.md`。
- HF：`binglingzhimeng/MiniMax-H3-b25-49-full`（b5/b20/b25/b30/b45 五份全量教师）。
