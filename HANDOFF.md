# ComfyUI-H3-ExactAdaLN 交接文档

> **换机后从这里开始。** 本文件自包含，新对话请先完整读完再动手。
> 所有结论都已实测或经源码核对，不要重复排查「已排除项」。
>
> 状态截至 `705a368`（2026-11-08）。`README.md` 是给用户看的用法说明，
> 本文件是给下一个接手的开发者看的状态、流程与坑。

---

## 0. 换机速查

| 项 | 值 |
|---|---|
| 仓库 | `https://github.com/bingling360/ComfyUI-H3-ExactAdaLN`（分支 `main`） |
| 代码状态 | `705a368`（b25-49 支持 + FL2VA 录制工作流）；此后仅本文档有更新 |
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
python tests/test_merge.py        # 纯 CPU，验证合并逻辑（需 torch/safetensors/numpy）
```

要跑真实模型，需要 ComfyUI 0.42.5+（实例上是 0.44.0）和「§7 模型清单」里的文件。

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

## 2. 节点清单（8 个）

| 节点 | 输入 → 输出 | 说明 |
|---|---|---|
| `H3AdaLNRecord` | full MODEL (+LoRA) → MODEL, TABLE | 采样期只读 hook，`time_embedder` 取 t，`adaln_proj` 取原生输出 |
| `H3ExactAdaLNBake` | full MODEL + sigmas → TABLE | 离线近似；支持 `splice` donor（plain float） |
| `H3ExactAdaLNSave` | TABLE → 文件路径 | 像 Save Image，自动编号；镜像到 `input/` |
| `H3ExactAdaLNLoad` | `table_name`（自由文本/子串/绝对路径）→ TABLE | 不用 COMBO（启动时缓存）也不支持上传控件 |
| `H3AdaLNMerge` | 两张 TABLE + `split` + `final_from` → TABLE | **bX-49 专用**：0..split-1 取 A，split..49 取 B |
| `H3ExactAdaLNApply` | pruned MODEL + TABLE → MODEL | `strict` / `fallback` / `off`；`off` 会先清理残留 hook |
| `H3LatentCompare` | 两个 LATENT → 报告 | 支持 NestedTensor；拒绝同源输入（防假 PASS） |
| `H3LatentSave` | LATENT → 文件 | 核心 `SaveLatent` 对 NestedTensor 会崩，必须用这个 |

---

## 3. 工作流清单（`workflows/`，全部由 `tools/build_*.py` 生成，不要手改）

| 文件 | 用途 |
|---|---|
| `01_bake_exact_adaln.json` | Bake → Save（离线，仅供快速看效果） |
| `02_verify_full_vs_pruned.json` | Load + Apply 的 A/B 验证（full vs pruned+表） |
| `03_verify_save_latent.json` | 同 02，多两个 `H3LatentSave` 落盘交叉检查 |
| `04_verify_strict.json` | 离线 Bake 路径的严格 A/B（**预期 FAIL，是根因证据**） |
| `05_record_strict.json` | **Record 路径的严格验证**：A 录 + B 注入，Compare 必须 PASS |
| `06_b25-49_hybrid.json` | 录 Ref2VA full + 与已有 FL2VA 表合并，一次运行产出 B 表与 hybrid 表 |
| `07_merge_tables.json` | 纯 CPU 重合并（两张表都已落盘时用） |
| `08_record_fl2va.json` | 录 FL2VA full 基准表（b25-49 的 0-24 半） |

生成命令（需要 ComfyUI 的 `blueprints/Image to Video (MiniMax H3).json`，
用 `COMFYUI_DIR` 或 `H3_BLUEPRINT` 指定）：

```bash
python tools/build_workflows.py   # 01, 02
python tools/add_savelatent.py    # 03（从 02）
python tools/build_strict.py      # 04
python tools/build_record.py      # 05
python tools/build_hybrid.py      # 06, 07, 08
```

工作流 JSON 在 `.gitattributes` 里强制 LF；在 Linux 上重生成后 `git status` 应为空。

---

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

## 6. 代码结构与关键实现

```
record_engine.py   采样期只读 hook（精确路径）
bake_engine.py     离线重算 + splice donor
apply_engine.py    运行时注入：t_emb 匹配 + sigma 兜底
sigma_plan.py      float32 精确的 t 运算，镜像 model.py:671-751
table_io.py        safetensors 容器 + merge_tables（bX-49 合并）
nodes.py           8 个节点
latent_io.py       H3LatentSave（NestedTensor）
workflows/         01-08（生成，不手改）
tests/             test_exact.py（21 项数值）/ test_workflows.py（结构）/ test_merge.py（10 项合并）
tools/             工作流生成器 + 模型/对照辅助
```

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
9. **表与 schedule 绑定**：换步数 / shift / 分辨率 / LoRA 强度都要重录；strict 会列出缺失 t。
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

### 11.7 尚未做

- b15-49 / b20-49 等其他切分的全量 teacher（`minimax-h3-other/` 里有对应的 pruned 版）。
- `tools/build_hybrid.py` 还没把 `time_embedder_from` 写进生成的 `06_b25-49_hybrid.json`。
- 模型卡里已写明「与老合并表不是同一目标」及全部实测数字。
