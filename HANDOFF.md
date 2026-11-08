# ComfyUI-H3-ExactAdaLN 修复交接文档

> **换机后从这三步开始**（状态截至 commit `f4065d9`，已完成 Phase 1–4）
>
> 1. `git pull`（或 `git clone git@github.com:bingling360/ComfyUI-H3-ExactAdaLN.git`）—— 代码、5 个工作流、两个测试脚本、工作流生成器都已在仓库里，本文档即仓库根的 `HANDOFF.md`。
> 2. 在实例上挂 `workflows/05_record_strict.json` 跑一次：`H3 Latent Compare` 必须是 `equal=True / max_abs=0 / VERDICT: PASS`。
> 3. 把 `H3 Exact AdaLN Apply` 的 `mode` 改成 `off` 再跑：必须 `FAIL`（对照组）。
>
> 结果与逐条落地方式见 **§7 执行结果**；本地已通过的 21 项数值/结构校验见 `tests/`。
> 若第 2 步仍 FAIL，先看 Compare 的 `max_abs`：`0` 附近但非 0 → dtype/算术路径；`1e-2` 量级 → 表没被注入（看 Apply 报告里的 `exact matches` 与日志里的 `first injection confirmed`）。

> 本文件自包含。新对话请先完整读完，再动手。所有结论均已实测或经源码核对，不要重复排查"已排除项"。

## 0. 项目基本信息

| 项 | 值 |
|---|---|
| 插件仓库（本地） | `D:/Adaln/ComfyUI-H3-ExactAdaLN/`（git 仓库，远端 `git@github.com:bingling360/ComfyUI-H3-ExactAdaLN.git`，分支 `main`） |
| 本地辅助生成器 | `D:/Adaln/tools/`（`build_workflows.py`、`build_strict.py`、`add_savelatent.py`、`cloud_fetch.sh`、`fetch_model.py`） |
| 工作流输出目录 | `D:/Adaln/workflows/`（仓库内副本 `ComfyUI-H3-ExactAdaLN/workflows/`） |
| ComfyUI 源码（本地，0.42.5） | `D:/ComfyUI_windows_portable/ComfyUI/` |
| 云端 ComfyUI | 0.44.0 / PyTorch 2.9.1+cu130 / RTX 4080 SUPER 32GB，源码在实例 `/root/ComfyUI/`，插件 `/root/ComfyUI/custom_nodes/ComfyUI-H3-ExactAdaLN/` |
| 当前 GitHub HEAD | `3e774d5` |

**用户最终目标**：在固定调度（或自定义 sigmas）下，让「剪枝版 MiniMax-H3 + 精确 AdaLN 表 + LoRA」的生成结果与「完整版 + 同一 LoRA」**逐位一致（bitwise）**，速度维持剪枝版水平；表要能 Save/Load 分享，并尽量适配 LoRA。

**用户操作约定**：只做本地开发 + 推 GitHub，不要把工作流或节点文件手动部署到云端实例；用户自己 pull / 挂工作流。

---

## 1. 已完成的事实（不要重复验证）

### 1.1 已跑通并确认的

- Bake 节点能在 ComfyUI 0.44.0 上从完整模型（`QuantizedTensor` 量化权重）烘焙出表：51 个投影（`blocks.0..49` + `final_layer`）、17 个 t 行、314.5 MiB、fp32 存储。
- Apply 节点确实在采样时注入了表：日志 `H3ExactAdaLN: first injection confirmed during sampling (blocks.0, 1 rows, indices=[0], max_dist=0.000e+00)`，8 步全程无 strict 报错。
- `H3LatentCompare` 已支持 0.44 的 `NestedTensor`（视频+音频双流），能展开 `samples__part0/1` 出逐张量明细。
- 严格验证工作流 `workflows/04_verify_strict.json`：同一次 prompt 内两条链共享 `RandomNoise`/`SIGMAS`/conditioning/`latent_image`，Bake 内存直连 Apply（无 Save/Load/上传），只用首发 3 个节点 + 核心节点，20 节点 27 连线，不缺节点。

### 1.2 已排除的差异来源（重要，别再查）

| 排除项 | 证据 |
|---|---|
| 跨次运行非确定性 | 同 prompt 双链仍 FAIL；且开/关 `--disable-comfy-compiler` 两次结果**逐位相同** |
| Comfy 编译器 / graph breaks | 加 `--disable-comfy-compiler` 后 `max_abs=3.556e+00 / mean_abs=1.449e-01 / PSNR=31.14 dB` 完全不变 |
| 完整/剪枝模型的非 AdaLN 权重不同 | 两文件 829 个共同非 AdaLN 张量 shape/dtype 零差异；6 个 73 MiB 量化大权重（`blocks.*.mlp.fc2.weight`）完整字节 SHA256 全 SAME；norm/bias/scale/`comfy_quant` 抽样也全 SAME。权重逐字节相同 |
| 表上传/Load/资产面板 | `H3ExactAdaLNLoad` 已改自由文本 `table_name`（文件名/子串/绝对路径），`H3ExactAdaLNSave` 已支持 NestedTensor；核心 `SaveLatent` 不兼容 H3 的 `NestedTensor`（没有 `.contiguous()`），已换掉 |
| 核心 `SaveLatent` 崩溃 | 已定位为 NestedTensor，插件自带 `H3LatentSave` 解决 |

### 1.3 严格验证的当前结果（**尚未达成目标**）

同 prompt 双链、关编译器前后一致：

```
samples__part0: equal=False max_abs=3.556e+00 mean_abs=1.449e-01 mse=5.301e-02 psnr=31.14 dB
samples__part1: equal=False max_abs=1.712e-01 mean_abs=1.901e-02 mse=6.989e-04 psnr=42.91 dB
compared tensor pairs: 2
VERDICT: FAIL - outputs differ
```

作为参照，完整模型**自己重跑两次**是 `video mean_abs=0.1685 / 29.11 dB`、`audio 0.0200 / 43.24 dB`。即剪枝+表的差异落在完整模型自身波动范围内，工程上已不可区分，但**不是 bitwise**。

---

## 2. 根因（实测 + 源码，已闭环）

### 2.1 决定性实测

在云端实例用 `comfy.sd.load_diffusion_model(FULL)` 加载完整模型，用表里 `t_values`（17 个 t）重算 `blocks.0.adaln_proj.linear(F.silu(time_embedder(t).to(bf16)))`，与表 `blocks.0` 比：

| 对比 | equal | max_abs | mean_abs |
|---|---|---|---|
| CPU 权重状态 vs 表 | False | 0.015625 | 6.66e-4 |
| CUDA 权重状态 vs 表 | False | 0.015625 | 5.24e-4 |
| CPU vs CUDA 互相 | False | 0.015625 | — |

`max_abs=0.015625` = 一个 bf16 ULP。**表值与完整模型原生 AdaLN 输出本来就差约 5e-4**，经 8 步扩散放大即 latent 的 0.145。权重是 `QuantizedTensor`（`device=cpu`），CPU/CUDA 走 `comfy_kitchen` 不同 backend（eager 反量化 vs CUDA 量化 GEMM），数值不同。

### 2.2 源码证据（本地 0.42.5 行号，0.44 对应逻辑相同）

| 位置 | 问题 |
|---|---|
| `ComfyUI-H3-ExactAdaLN/bake_engine.py:130/134` | **最根因**：在模型未经 `load_models_gpu`/`patch_model` 的状态下直接调 `dm.time_embedder(t_vals)` / `ap.linear(inp)`。Bake 测的是 CPU 原始权重 + eager 路径；采样时是 GPU 已 load/已 patch + 量化 GEMM |
| `comfy/model_patcher.py:921-922` | `if key not in self.patches and not force_cast: return weight` —— 无 LoRA 的量化权重根本不反量化，"物化有效权重"承诺落空 |
| `bake_engine.py:74` | `patch_weight_to_device(key, inplace_update=True)` 撞 `comfy/ops.py:1564` 的 `assert inplace_update is False`；量化 teacher + AdaLN LoRA 会失败（被 errors 吞掉）。`ops.py:1558` 还会 `requantize_from_float(scale="recalculate", stochastic_rounding=...)` |
| `bake_engine.py:116/140` | `clone()` 共享底层 `nn.Module` 和 backup dict（`model_patcher.py:429-430,447,460`）；`unpatch_model()` 污染原 ModelPatcher，`CoreModelPatcher.unpatch_model`(`model_patcher.py:2159`) 还跳过基类恢复块 |
| `ComfyUI-H3-ExactAdaLN/sigma_plan.py:78` | `round(float(t), 7)`，而 `comfy/ldm/minimax/model.py:743` 用未 round 的 `unique_t`；`TimeEmbedder` 的 `exp(-log(10000)*k/128)`(`model.py:143`) 放大误差 |
| `ComfyUI-H3-ExactAdaLN/apply_engine.py:164` | 表转 `out_dtype=bf16` 注入，但 curve 模型 `adaln_dtype=torch.float32`(`model.py:496`)；`FinalLayer.mod`(`model.py:322`) 用 `scale.dtype` 决定乘加精度 → 不等价 |
| `apply_engine.py:66/71` | curve keys 在 CPU、用 round 后的 t；运行时 `model.py:746` 用运行时 device + 未 round 的 t，lerp 公式一致但非逐位一致 |
| `apply_engine.py:102-103,179` | `tol = max(1e-3, 2e-3*scale)` 偏宽松，只按 `maxd` 整体判定，t≈0.999/1.0 附近可能静默错行 |
| `apply_engine.py:242-244` | `use_match=True` 时不注册 dit patch 兜底；假想"逐块混搭"模型下会静默半注入且 `summary()` 不反映 |

**形状/chunk 是对的**：`apply_engine.py:218-220` 的 `reshape(m*modalities, expand*hidden).chunk(expand)` 与 `model.py:227-228` 等价。问题全在数值来源与 dtype。

**一句话**：靠"离线独立 Bake"永远复现不了采样时的量化数值状态，因此拿不到 bitwise。必须改成**记录式**。

---

## 3. 另外两个调研结论

### 3.1 b25-49 混合模型

- **不是「full 块 + pruned 块」拼接**：是「以 FL2VA 为基底，把尾部 25 块的 `adaln_proj.linear.weight/bias` 换成 Ref2VA」的张量级权重选择（证据：`ComfyUI-minimaxH3-SequenceForge/README.md:184-202`、HF `jfar-z/MiniMax-H3-FL2VA-Ref2VA-Hybrid-NVFP4` 的 Merge Method）。`b15-49/b20-49/b25-49/b30-49` 是 0-based 闭区间替换块范围。
- 基于 pruned base 的 hybrid 整个文件都是 curve 形态，`use_adaln_curves` 是**全局**（`model.py:493`），51 块全 8 维；`model_detection.py:418-428` 只靠 `adaln_t_table` key 判形态，看不见 hybrid。
- **表技术上可用**（存的是最终调制值，与架构无关），`_build_match_keys` 在真 hybrid 上不会维度失配。
- **但语义会错**：FL2VA full teacher 的表把 51 块全替换成 FL2VA 调制值，会抹掉 hybrid 尾部 ref2va 的参考条件能力。要正确支持，Bake 需加 `splice` 输入（照 HyperFlow `Create Curve Fit` 的 `FILE:FIRST-LAST` 格式，如 `...ref2va...safetensors:25-49`），烘焙时把指定区间 AdaLN 权重换成 donor 再算表，meta 记 splice 指纹。

### 3.2 sigmas shift 不是必须的

- **严格相等的充要条件 = 「运行时所有 t 值 ⊆ 表的 t 集合」+「t→t_emb 映射一致」**；`shift_v/shift_a` 参数相同既不充分也不必要。
  - 不充分：shift 相同但 Bake 8 步 simple、运行时 10 步 ManualSigmas，t 集合不同照样失败。
  - 不必要：把运行时 `SIGMAS` 接进 Bake 的 `sigmas` 输入，或用 `extra_t` 补齐实际 t，就能全命中。
- ManualSigmas 绕开了 shift 对 sigma 网格的影响，但 `t_a = 1-time_shift_sigma(σ,shift_v,shift_a)`(`model.py:676`) 和 `audio_scale = shift/audio_shift`(`model_sampling.py:353`) 仍读 shift。
- `ModelSamplingMiniMaxH3`(`nodes_minimax_h3.py:392-408`) 同时写 `set_parameters` 和 `transformer_options` 两处；`_auto_shifts` 只读 `model_sampling`，可能与 `model.py:672-673` 不一致。

---

## 4. 修复计划

### Phase 0 — 准备（不改逻辑）

1. 读本文件 + `README.md` + 6 个 `.py`：`nodes.py`、`bake_engine.py`、`apply_engine.py`、`sigma_plan.py`、`table_io.py`、`latent_io.py`。
2. 确认本地能跑静态测试：`D:/ComfyUI_windows_portable/python_embeded/python.exe` + `sys.path` 加 ComfyUI 根和插件目录；`py_compile` 全部 `.py`。
3. 保持 `git` 干净，每个 Phase 单独 commit + push，`git -c gc.auto=0 -c maintenance.auto=false -C ... push origin main`。

### Phase 1 — 消除确定性数值偏差（最高优先，独立可验收）

**目标**：让表值 == 完整模型在采样状态下的原生 AdaLN 输出（逐位）。

1. **新增 record 模式（核心）**，二选一实现：
   - **推荐 A**：新增节点 `H3AdaLNRecord`（或给 `H3ExactAdaLNBake` 加 `mode="record"`）。它挂 51 个 `adaln_proj` 的**只读** forward hook，在**真实采样过程中**把每次原生输出按 t 记录：
     - hook 拿到 `output`（`tuple` of 6 张表，每张 `[M*modalities, hidden]`）→ `torch.cat(output, dim=-1)` 得 `[M*modalities, expand*hidden]`，`reshape(M, 96768)` 得 raw 行；
     - t 索引来源：用 `WrappersMP.DIFFUSION_MODEL` wrapper（参照 `model.py:623-628`）在 `_forward` 前从 `transformer_options["sigmas"]` + `sigma_plan.runtime_unique_t()` 算出本步 `unique_t`，存进 runtime state；**不要依赖 `patches_replace`**（SequenceForge 直接调 `comfy.sample.sample()` 时不触发）。
     - 新 t 才写入，重复 t 校验一致性；跑完输出 `H3_ADALN_TABLE` → 接 `H3ExactAdaLNSave`。
   - **备选 B**：修 `bake_engine.bake`，在算表前显式 `comfy.model_management.load_models_gpu([mp])` / `mp.load(device_to=...)`，并用 `patch_weight_to_device(key, device_to=<同一 device>, force_cast=True, inplace_update=False, return_weight=True)` 取有效权重后自己 `F.linear`。风险：仍可能与采样时 backend 不同，**不如 A 彻底**。
2. 去掉 `sigma_plan.py:78` 的 `round(..., 7)`，让表 t 与 `model.py:709` 的 `unique_t` 用同一组未舍入 float（保留 `build_index` 的小容差匹配）。
3. `out_dtype` 跟随目标模型：curve 用 `torch.float32`（`model.py:496`），full 用 bf16；至少保证 `FinalLayer` 的 `scale.dtype` 与原生一致。
4. `_build_match_keys` 的 curve 分支改为在**运行时 device**、用**未 round 的 t** 计算；或更好：把 `[T,k]` key 矩阵随表保存（`table_io` 增加可选张量），Apply 直接用，与 Apply 端形态彻底解耦。
5. 收紧 `_match_indices`：逐行校验距离、缩小 `tol`、错行时明确报错；`use_match=True` 时也注册 dit 兜底。

**验收**：在实例上重跑 §2.1 的独立脚本，`record` 出来的表 vs 完整模型原生输出应为 `equal=True / max_abs=0`（CPU 与 CUDA 各验一次）。这一步不跑完整采样也能验。

### Phase 2 — 端到端 bitwise 验证

1. 用 `04_verify_strict.json`（同 prompt 双链）改造：A 链挂 `H3AdaLNRecord`（只读），B 链 `Apply` 从**同一次执行**里拿到记录的表。
   - 顺序问题：ComfyUI 按拓扑并行执行两个采样器。需让 B 链采样器的某个输入（例如 `latent_image`）经过一个 pass-through 节点依赖 A 链 latent 输出，强制 A 先完成、表填满，再跑 B。
   - 备选：分两张工作流（Step1 A 链 record + Save；Step2 同 prompt 双链，B 链 Load 表）。因为 AdaLN 只依赖 t、与 noise 无关，跨次记录不影响表值；只要 t 集合一致即可。
2. 验收：`H3LatentCompare` 报告 `samples__part0/1: equal=True max_abs=0`，`VERDICT: PASS - bitwise identical`。
3. 对照：Apply `mode=off` 再跑，必须 `FAIL`。

### Phase 3 — 混合模型与 t 集合校验

1. `H3ExactAdaLNBake` 加 `splice` 字符串输入（`FILE:FIRST-LAST`），烘焙时按区间把 `adaln_proj.linear.{weight,bias}` 换成 donor，再算表；meta 记 splice + donor 指纹，Apply 的 fingerprint 校验扩展包含它。
2. `ExactAdaLNRuntime.match_keys` 从单矩阵改成 `{dim: [T,dim]}` 字典，`register()` 读每个 `ap.linear.in_features` 绑定对应 key 集；`make_hook(key, keyset)` 去掉全局维度判断。
3. Apply 改用「t 集合覆盖」校验：运行时算 `runtime_unique_t()` 与 `table.t_values` 做集合差，缺哪几行直接列出并拒绝，而不是等 `cdist` 失败。报错文案从 "different sigma schedule / shift" 改成 "t set mismatch — 某些 t 缺失"。
4. `_auto_shifts` 优先读 `model_options["transformer_options"]["minimax_h3_sigma_shift_*"]`，回退 `dm.sigma_shift_*`，最后 `model_sampling`，与 `model.py:672-673` 对齐。
5. 更新 README：Known limits、Status、混合模型章节、t 集合覆盖口径。

### Phase 4 — 收尾

1. 重新生成/校验所有工作流（`python tools/build_workflows.py`、`build_strict.py`），确保连线类型一致、无缺节点。
2. 本地静态测试全过（节点 schema、表往返、NestedTensor 比对、record hook 数值、工作流 27 条连线）。
3. commit + push；**不要部署到实例**，告诉用户 pull 即可。

---

## 5. 关键坑（新对话务必注意）

- 本机 IPv6 黑洞：所有网络操作强制 IPv4（`curl -4`、Python patch `socket.getaddrinfo`），并 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`。GitHub HTTPS 常不通，但 `git@github.com` SSH 通；推送用 `git -c gc.auto=0 -c maintenance.auto=false -C <repo> push origin main`。
- 本机 Git Bash 缺部分 coreutils（`mkdir`/`head`/`grep` 等可能不可用），建目录/写文件用 Python 脚本。
- 云端 ComfyUI 由 `/root/restart.sh` 启动（`nohup python main.py --listen 0.0.0.0 > comfyUI.log 2>&1 &` + `tail -f`）。长 SSH 命令容易卡住，用短连接 + 远程轮询；不要在同一条 SSH 里同时 `nohup` 重启和长轮询。
- `comfy.sd.load_diffusion_model(path, model_options={}, disable_dynamic=False)` 返回 `ModelPatcher`，权重默认停在 CPU/offload，**必须显式 load 才是采样时状态**。
- `H3LatentCompare` / `H3LatentSave` 必须处理 `comfy.nested_tensor.NestedTensor`（`.is_nested` + `.unbind()`），0.44 的 H3 LATENT 是 `{"samples": NestedTensor}`，不是普通 Tensor。
- 核心 `SaveLatent` 对 NestedTensor 会崩（`nodes.py:558` `.contiguous()`），只能用插件的 `H3LatentSave`。
- `H3ExactAdaLNLoad` 用自由文本 `table_name` + `table_path`；不要用 COMBO + `image_upload`（前者启动时缓存、后者只收图片格式）。
- 表的 `out_dtype` 与 curve 模型 `adaln_dtype=fp32` 的关系是确定性偏差来源之一。
- 用户已明确：**不要手动部署工作流/节点到实例**，只推 GitHub。

---

## 6. 验收标准（最终）

1. `04_verify_strict.json`（或 Phase 2 改造版）跑一次，Compare 报告 `equal=True`、`max_abs=0`、`VERDICT: PASS - bitwise identical`。
2. Apply `mode=off` 对照必须为 `FAIL`。
3. 同一张表可用于 full / curve / b25-49 hybrid（后者需 splice 正确），t 集合覆盖校验能列出缺失 t。
4. 本地静态测试全过，仓库工作区干净，远端 HEAD = 本地 HEAD。

---

## 7. 执行结果（已完成，`f4065d9`）

Phase 1–4 全部落地并推送。核心变化：**放弃"离线独立 Bake"，改为采样期记录。**

| Plan 条目 | 落地方式 |
| --- | --- |
| Phase 1.1 record 模式 | 新模块 `record_engine.py` + 新节点 `H3AdaLNRecord`。hook `time_embedder` 直接拿 `t_vals`（行↔t 一一对应，不做任何重建），hook `adaln_proj.linear` 拿 raw 输出。**备选 B（修 bake 的 load/patch）没走**——它仍然依赖"手工选 kernel"，原理上拿不到逐位。 |
| Phase 1.2 去 round | `sigma_plan.f32_image()` 存 t 的 float32 像，`bake_t_values` 精确去重，`round(t,7)` 删除 |
| Phase 1.3 out_dtype | 改为**实测记录**（`recorder.out_dtype`）或 `teacher_out_dtype(dm)`（按 model.py:495 推），不再由 `compute_dtype` 下拉决定 |
| Phase 1.4 运行时 device/未 round t | `_keys_for()` 懒构建，用 `t_emb` 自己的 device + dtype；`[T,k]` 随表保存改成**按 t_emb 宽度分发**（`{width: kind}`），更彻底 |
| Phase 1.5 收紧匹配 | 容差 `1e-6*|keys|`（远小于 1 个 bf16 ulp），失败时列出缺哪些 t；`use_match=True` 时也有 `WrappersMP.DIFFUSION_MODEL` 兜底 |
| Phase 2 端到端 | `workflows/05_record_strict.json`：A 链 record、B 链 apply，`after` 通配输入强制 A 先跑；Save 也接 `after` |
| Phase 3.1 splice | `H3ExactAdaLNBake.splice = FILE:FIRST-LAST`，donor 必须是 plain float checkpoint，meta 记指纹 |
| Phase 3.2/3.3 | 见上（按宽度分发 + t 集合覆盖报错） |
| Phase 3.4 `_auto_shifts` | transformer_options → `dm.sigma_shift_*` → `model_sampling`，与 model.py:672 一致 |
| Phase 4 | 5 个工作流重生成 + `tests/test_workflows.py` 结构校验；`tests/test_exact.py` 21 项数值校验全过；工作区干净，远端 HEAD = 本地 HEAD |

**已本地证明的结论**（`tests/test_exact.py`，真实 `MiniMaxH3Model` 微型配置）：
完整 teacher 与 curve student 共享全部非 AdaLN 权重，record 4 步后 apply，
4/4 步 `torch.equal`、`max_dist=0.000e+00`、12 次精确 t_emb 命中；
对照组（不注入）`max_abs=3.8e-01`。

**实例验收（2026-11-08，已完成）**：ComfyUI 0.44.0 / RTX 4080 SUPER 上用同一个
API prompt（从 `/history` 取回）连跑三次 `05_record_strict.json`：

| 次序 | Apply `mode` | Compare | 注入日志 |
|---|---|---|---|
| 1 | `strict` | `PASS / max_abs=0`（90.83s） | `first injection confirmed` |
| 2 | `off` | `FAIL / max_abs=3.349e+00`（video）、`2.970e-01`（audio） | 无 —— 上一次的 hook 已被清理 |
| 3 | `strict`（off 之后） | `PASS / max_abs=0` | `first injection confirmed` |

报告落盘 `output/h3_adaln/compare_*_145323 / 145341 / 145405.txt`；修改前的首次
PASS 日志备份在 `/root/comfyUI_pre_fix_PASS_20261108.log`。第 2/3 次证明「残留 hook
导致对照组假 PASS」已修掉，且清理不会误伤后续 strict 运行。`04` 保留为 Bake
离线路径的 A/B 对照（预期仍 FAIL，这正是根因证据）。

**修复提交**：`274490e`（off 清理残留 hook、LoRA 指纹纳入权重与强度、hook 按
共享 `nn.Module` 跟踪、Compare 拒绝同源输入、工作流 `widgets_values_named` /
`properties.models` 同步、0 维 t 防崩）、`da62ef2`（工作流 JSON 统一 LF）。
本地与实例均为 `da62ef2`；`tools/build_*.py` 重生成后 `git status` 为空
（byte-for-byte）；`tests/test_workflows.py` 与 `tests/test_exact.py`（21 项）全过。
5 个工作流已部署到实例 `user/default/workflows/`，ComfyUI 已重启加载新节点。
