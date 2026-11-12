# CANN Bench 实测：agent 在“高难度算子”上的表现（第二步）

日期：2026-11-12
设备：m18 `kylin10-6018` / Ascend910B4-1（`ASCEND_RT_VISIBLE_DEVICES=1`）
环境：容器 `deepinfer-real`，CANN 9.1.0，torch 2.10.0+cpu，torch_npu 2.10.0.post4
模型：`deepseek-chat`（DeepSeek OpenAI 兼容端点）
测量：官方风格正确性用例（8 例）+ 官方 bench（8 个大 shape，3 轮取中位数，95% bootstrap CI）

## 这一轮要回答的问题

第一步（`20261108-eval.md`）测的是 add / swi_glu / rms_norm 这类单算子。结论是“正确、可运行，但没有稳定超过 vendor 脚手架”。
本轮换成两个**高难度**算子：`kv_cache_append`（带历史缓冲、多维拷贝）和 `moe_gating_top_k_softmax`（softmax + top-k + 行索引）。
先想测 `grouped_matmul` 和 `moe_re_routing`，但脚手架里这两个不可用：

- `grouped_matmul`：只有 Python 包装，没有 C++ op / 注册，无法编译测试。
- `moe_re_routing`：vendor kernel 在 CANN 9.1 上编译失败（`Adds` 对 `__bf16` / `signed char` 的 static_assert），不能作为基线。

所以换成上面两个。

## 结果

| 任务 | 官方正确性 | vendor 速度比（95% CI） | agent 最终解（95% CI） | agent / vendor | 最差用例 |
| --- | --- | --- | --- | --- | --- |
| `kv_cache_append` | vendor 8/8；agent 8/8 | **0.0771** [0.0698, 0.0781] | **1.2334** [1.2236, 1.2754] | **16.0x** | 0.3995 |
| `moe_gating_top_k_softmax` | vendor 8/8；agent 8/8 | **0.0903** [0.0871, 0.0904] | **0.2424** [0.2399, 0.2437] | **2.68x** | 0.0818 |

速度比 = 8 个 bench 用例的几何平均（candidate vs torch eager），3 轮 bench 的中位数，CI 为 bootstrap 95%。
`total_speedup`（总时长比）：kv 0.0729 → 1.0392；moe 0.0921 → 0.2704。

### kv_cache_append 逐用例（agent 最终解）

| shape (B,S,H,D,T) | dtype | speedup |
| --- | --- | --- |
| (1,1024,8,128,128) | float16 | 2.8535 |
| (1,2048,8,128,256) | float16 | 1.5067 |
| (2,1024,8,128,256) | bfloat16 | 1.5121 |
| (1,4096,4,128,512) | bfloat16 | 0.7969 |
| (8,512,8,64,512) | float16 | 0.8119 |
| (4,1024,16,64,256) | float16 | 1.4858 |
| (1,8192,2,128,1024) | bfloat16 | 0.3995 |
| (2,2048,8,64,128) | bfloat16 | 2.8050 |

agent 把 vendor 的“整块历史拷贝”换成了融合的 `launch_kv_copy2_*` + k/v 融合写入，整体从 0.077x 提到 1.23x；
但 3 个 shape 仍低于 1.0（0.80 / 0.81 / 0.40），说明收益集中在 H 大、拷贝占比高的形状，D 大 / S 大的形状还有退化。

### moe_gating_top_k_softmax 逐用例（agent 最终解）

| shape (R,C,k) | dtype | speedup |
| --- | --- | --- |
| (4096,64,2) | float16 | 0.5348 |
| (8192,64,4) | float16 | 0.3565 |
| (16384,128,6) | bfloat16 | 0.1664 |
| (4096,256,8) | bfloat16 | 0.1380 |
| (32768,32,2) | bfloat16 | 0.4009 |
| (2048,512,4) | float16 | 0.2087 |
| (65536,16,2) | float16 | 0.3974 |
| (8192,128,8) | float32 | 0.0818 |

agent 修好了 vendor 的 tiling 并把 kernel 提速 2.7x，但**所有形状仍慢于 torch**（最好 0.53x，最差 0.08x）。
这是一个“能改进、但远未追上 eager”的典型：top-k 的选择和行索引生成用 AscendC 写出来，代价远高于 torch 的融合实现。

## 结论

- **简单/中等算子**：agent 能写出正确实现，最好能超过 vendor（add +28%），其余与 vendor 同档。
- **高难度算子**：agent 能显著改进一个很差的 vendor 实现（kv 16x、moe 2.7x），但只在 kv 上跨过了 “快于 torch” 这条线，
  且形状间差异很大（kv 0.40–2.85，moe 0.08–0.53）。**“高等级算子”目前还不行**，与第一步的判断一致。
- vendor 在这两个算子上的基线本身是参考实现级别（0.077x / 0.090x torch），因此 “agent / vendor” 会显著夸大实际水平；
  对外应同时给出 vs torch 的绝对数字。

## 本轮修掉的评测器问题

这些问题会让“agent 失败”看起来像环境问题，或把 vendor 判错，必须记录：

1. **`run_cannbench` 把 bench 的非零退出码一律映射成 `bench_error`。**
   bench 脚本在候选输出错误时按设计返回 1，但 metrics 已经完整打印；旧代码先看退出码、后读 metrics，
   于是 agent 只看到 “bench exited 1”，16 次迭代全部 `bench_error`。
   现在先读 metrics，失败用例折成 `fail_correctness` 并在 error 里列出 shape/dtype（PR #11）。
2. **`bench_nonce.py` 在非零退出时提前退出，metrics 没写文件。** 即使修了 (1)，没有 payload 仍然无法判定；现在 payload 优先。
3. **测试/bench 脚本没有 `import torch_npu`。** `torch.device("npu")` 在部分进程里直接抛
   `ModuleNotFoundError: No module named 'torch.npu'`，agent 的第二次运行 8 次迭代全部在算子执行前失败。
   四个脚本已显式 import。
4. **moe bench 的参考值有两处自身错误**：行索引用 NPU 上的 `arange(int32)` 生成（某些尺寸返回垃圾值）；
   参考张量在 vendor op 启动前没有 sync（op 走 side stream，参考值有时被读成 0）。修正后 vendor 为 8/8。
   之前记录的 “moe vendor 2/8 不正确” 是我方 bench 的 bug，不是 vendor 的缺陷——此处更正。

## 限制

- 只测了 `deepseek-chat`，每个任务只跑了 1 次；m18 是共享机器，CI 用同任务、同设备的 vendor/agent 对照降低干扰。
- bench 的 8 个大 shape 与官方正确性用例的 8 个小 shape 不同：正确性只在小 shape 上被官方用例覆盖，
  大 shape 的正确性由 bench 自带检查覆盖（agent 最终解两处都是 8/8）。
- 这不是对 910B 的性能承诺，是单次实测。

## 原始结果（m18）

- vendor：`/data2/kernel-agent/cb_results/vendor_kv_3r.json`、`moe_vendor_3r.json`
- agent 最终解：`/data2/kernel-agent/cb_results/agent_kv_final3.json`、`agent_moe_final3.json`
- 运行记录：`/data2/kernel-agent/cb_eval/kv_runs/20261111-084359_kv_cache_append_910b/`、
  `/data2/kernel-agent/cb_eval/moegate_runs3/20261112-085407_moe_gating_top_k_softmax_910b/`
- 评测脚本（在 agent 工作区之外，agent 不可修改）：`/data2/kernel-agent/cb_tools/`