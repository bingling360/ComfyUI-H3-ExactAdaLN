# ComfyUI-H3-ExactAdaLN

> Make a **pruned** MiniMax-H3 checkpoint reproduce the **full** model's AdaLN modulation bitwise.

让剪枝版 MiniMax-H3 逐位复刻完整模型的 AdaLN 调制：**剪枝基座 + 表 + LoRA ≡ 完整教师 + 同一 LoRA**（逐位相同），速度维持剪枝版水平。表自动落盘、自包含，可直接拷贝分享。

## 一、原理

H3 的 AdaLN 分支输出只依赖 timestep：`modulation = adaln_proj(time_embedder(t))`。Comfy-Org 的剪枝检查点把这条分支折成 8 维曲线基（`adaln_t_table`），于是：

- 曲线行之间需要插值、存储精度发生变化，不再是教师的原值；
- `time_embedder` 消失后，作用在 `time_embedder` / `adaln_proj` 上的 LoRA 会被 ComfyUI 直接丢弃，剪枝版等于丢掉了这部分 LoRA 效果。

本插件不拟合曲线，而是**保存最终调制值**：

1. **录制**：在完整教师 + 实际使用的 LoRA 上跑一次真实采样，只读前向 hook 捕获每个 timestep 上全部 51 个 `adaln_proj.linear` 的真实输出，按 t 的精确 float32 值存成表（fp32 存储）。捕的是模型自己算出来的张量，所以 LoRA、量化、运行期 kernel 全都已经在里面 —— 逐位一致是构造出来的，不是拟合出来的。
2. **应用**：在剪枝基座上按 t 精确命中表行并替换投影输出。命中不了就明确报错，绝不吸附到相邻行（那会造成静默的一行偏移）。
3. **表自带归属**：safetensors header 记录教师指纹、AdaLN LoRA 指纹、compute dtype、t 行集合，以及 `target_base`（目标剪枝基座的 AdaLN 指纹）。`table=(auto)` 靠这个指纹确定性查找：不猜、不看「最近一张」、不依赖任何外部状态文件。表改名、拷到别的机器都能用。

离线重算（v1 的 Bake 节点）已移除：在 CPU 权重上重算复现不了采样时的 GPU kernel，会带来 bf16 级误差并被多步放大。与「把 LoRA 投回曲线基」的同类方案相比，这里存的是最终值而非近似投影，代价是表更大、且与录过的 schedule 绑定。

## 二、功能

| 节点 | 输入 | 作用 |
|---|---|---|
| `H3 AdaLN` | `model`, `mode` (`apply` / `record` / `off`), `for_model`, `table`, 可选 `table_in` / `after` | 录制表、注入表、直通对照，三合一 |
| `H3 AdaLN Hybrid` | `model`, `table_a`, `table_b`, `split` | bX-49 混合教师：按块拼接两张表（仅这一种模型用） |
| `H3 Latent Compare` | 两个 LATENT | 输出 `equal` / `max_abs` / `PSNR`，验证是否逐位一致 |

表的行为：

- **自动命名与合并**：文件名由「教师指纹 + AdaLN LoRA 指纹 + dtype」算出；重复录制同一 schedule 不改动，换 schedule 只**追加**缺失的 t 行（不新建表）；只有同一个 t 出现不同数值（模型 / LoRA / dtype 变了）才拒绝写入。
- **一张表覆盖所有采样段**：调制只与 t 有关，与分辨率、帧数、seed、参考图、关键帧、pass 都无关。`fill_reference_row`（默认开）会顺带预计算参考图 / 参考音频要用的固定行 `t=0.999` / `1.0`。
- **自包含**：来源信息全在 header 里，`ComfyUI/models/h3_exact_adaln/` 中的 `.safetensors` 直接复制即可分享。

## 三、如何使用

### 安装

```bash
cd ComfyUI/custom_nodes
git clone --depth 1 -b main https://github.com/bingling360/ComfyUI-H3-ExactAdaLN.git
```

国内网络 clone 不通时可用 `https://gh-proxy.com/https://github.com/...` 前缀，或从 codeload 下载 ZIP 解压到同名目录。重启 ComfyUI。要求 ComfyUI 0.42.5+、PyTorch、`safetensors`，无需额外 pip 包。

### 三步

1. **录制**：加载**完整教师** + 你的 LoRA，把 `H3 AdaLN` 插在 shift 节点与 guider 之间，`mode=record`，在 `for_model` 下拉里**选中你准备使用的那个剪枝基座**，然后跑一遍你实际要用的工作流。prompt 结束时表自动写入 `models/h3_exact_adaln/`，不需要 Save 节点。
   - `for_model` 没有 auto 模式：目录里有多个相似基座时，只有你知道要用哪个，节点不扫描、不记忆、不猜测。这个选择会随 workflow 保存，**只需设置一次**；它的指纹写进表头后，之后所有推理都不用再选。
   - 录制可以用最小尺寸（H3 latent 支持 `256x128` / 5 帧）来省时间：表只依赖 t，与分辨率、帧数无关。但步数、scheduler、shift、参考 / 关键帧配置必须与实际出片一致。
2. **应用**：换成该剪枝基座，`mode=apply`，`table` 保持 `(auto)`。表头的 `target_base` 会命中当前模型，无需选表。
   - 同一基座录过多张表（不同 AdaLN LoRA 或同 LoRA 不同强度）时，会先按当前加载的 AdaLN LoRA 指纹（含 strength）筛出对应那张，再匹配 `target_base`；没录过的 LoRA / 强度直接报错，不会静默用错。
3. **验证（可选）**：同一张图把 `mode` 改成 `off` 再跑，latent 必须明显不同；用 `H3 Latent Compare` 对照应得到 `VERDICT: PASS - bitwise identical`。

同一个 prompt 里录完立刻用（A/B 验证）时，把另一条链的 LATENT 接到 `after`（仅用于排序），或用 `table_in` 串表；日常跨 prompt 使用不需要。

`workflows/` 下有两份 **API 格式** prompt（POST 到 `/prompt`，不是 UI 工作流导出）：`v2_record_apply_ab.json`（单 prompt 双链 A/B 验证）、`v2_apply_auto.json`（跨 prompt 自动查表）。使用前把里面的 checkpoint / LoRA 名改成你自己的。

## 四、注意事项

**表的身份与边界**

- 表 = （教师 AdaLN 权重、**AdaLN / timestep LoRA 及其强度**、compute dtype）+ 录制过的 t 集合。其他 LoRA（注意力、MLP、norm）不影响调制，**一张表通吃**。
- 碰 `time_embedder` / `adaln_proj` 的 LoRA 会被烘焙进表里（这正是插件的关键能力）；换 LoRA 或改强度就是另一张表，apply 会校验指纹并拒绝混用。强度不能运行时插值（两次 SiLU 非线性），要改就重录。
- full-form 的 `adaln_proj` LoRA（in-width 2688）在剪枝基座上会被 ComfyUI 丢弃，**必须挂在完整教师上录制**才能保住它的 AdaLN 效果。
- **只有换 steps / scheduler / shift 才是在同一张表上追加行**（旧行仍正确，再录一次补上缺失的 t，不覆盖）；换 AdaLN LoRA / 改强度是**新建一张表**：旧表保留、两套并存、按当前 LoRA 指纹自动分流，且新表不继承旧表为其他 schedule 积累的 t 行 —— 新 LoRA 下要用几套 schedule，就得各录一次。t 精确匹配（float32 位模式），不吸附近似值。
- bf16 录的表和 fp32 录的表是两张表；apply 发现 dtype 不符会警告 `replay is NOT bitwise`。

**录制与运行**

- `record` 必须在**完整教师**上跑：剪枝基座没有 `time_embedder`，节点会直接报错。录制需要一次完整采样，峰值显存按教师算；表本身约 315 MiB / 17 个 t 行，随录过的 schedule 增长。
- 没有声明 `target_base` 的表（没设 `for_model`、或 v1 旧表）`table=(auto)` **不会**使用：宁可报错也不静默用错教师。请在 `table` 里显式指定一次，或重录时补上 `for_model`。
- 同一个基座被两张表声明、或一张表都不匹配，都会明确报错并列出表头的 `target_base` 与 LoRA 指纹；`mode=off` 可显式不用表。
- 不支持逐行 `denoise_mask` 的 timestep、HyperFlow 式 `(t, r)` 端点条件，以及绕过 `module.forward` 的 C++ / 融合 kernel。
- bX-49 类混合模型：剪枝文件本身是「基座主体 + 捐赠者尾部」，无法直接在它上面录制；请在**完整的 bX-49 教师**上录一次（现成权重：`binglingzhimeng/MiniMax-H3-b25-49-full`）。`H3 AdaLN Hybrid` 拼接两张独立录的表只是近似路径，不是逐位解。
- ComfyUI 缓存会制造假 PASS：换模型 / 换表 / 换 mode 后换 seed 或重启再对照。

**附**：详细参数与排错见 [docs/NODE_GUIDE.md](docs/NODE_GUIDE.md)，设计说明见 [docs/unified-node.md](docs/unified-node.md)，[HANDOFF.md](HANDOFF.md) 是历史验证记录（节点行为以本文为准）。CPU 测试：`python tests/test_simple.py`、`tests/test_workflows.py`、`tests/test_merge.py`；`test_exact.py` / `test_runtime.py` 需要 ComfyUI 环境（`COMFYUI_DIR`）。
