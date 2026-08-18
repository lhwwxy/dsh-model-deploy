# dsh-model-deploy

> 🌐 语言切换 / Language: **[简体中文](./README.md)** · **[English](./README.en.md)**

DeepSeek Harness（DSH）插件：**LLM 模型选型部署分析**。

给它模型、GPU/NPU 型号、节点数、精度、上下文长度、并发数与互联带宽，它用一阶解析模型给出可审计的估算：

- ✅/⚠️/❌ **能否部署**（显存/精度/互联检查）+ 修复建议
- **显存**：单卡与集群拆解（权重 / KV 缓存 / 激活 / 运行时）
- **时延**：首 token（TTFT）与整体时延（空闲 / 满载两种口径）
- **吞吐**：单请求与系统 tok/s、req/s、预填充 tok/s
- **功耗**：空闲 / 典型 / 峰值（含 PUE）、每 token 能耗
- 自动 **TP × PP × DP** 并行策略与选择理由


## 安装

```sh
dsh plugin --profile web add dsh-model-deploy
```

或从 GitHub 检出安装：

```sh
dsh plugin --profile web add github:lhwwxy/dsh-model-deploy
```

> `--profile` 必填。安装后重启会话（或 profile），工具 schema 才会进入 prompt 组装。

## 工具

### `model_deploy_analyze`

主分析工具。示例调用：

```json
{
  "model": "deepseek-v3",
  "gpu": "ascend910b",
  "gpusPerNode": 8,
  "nodes": 1,
  "precision": "int4",
  "ctx": 131072,
  "batch": 4
}
```

字段：`model`（目录 id，或自定义模型 JSON 字符串）、`gpu`（目录 id）、`gpusPerNode`
(1–8)、`nodes` (1–64)、`precision`（`fp32|bf16|fp16|fp8|int8|int4|fp4`）、`ctx`/`batch`/
`prompt`/`output`（tokens）、`intraMode`（`auto|custom`）+ `intraGBs`、`interMode`
（`none|ib400|ib800|roce100|roce200|custom`）+ `interGBs`。

返回结构化报告（`status`、`issues`、`strategy`、`memory`、`performance`、`power`、
`assumptions`），渲染输出为简洁中文文本报告。

### `model_deploy_catalog`

列出支持的模型（id、参数量、层数、上下文上限）与 GPU/NPU（id、显存、带宽、
FP16/FP8/INT8/FP4、互联、TDP），支持 `query`/`vendor` 过滤。先用它查 id，再调分析工具。

## 覆盖范围

- **38 个模型** — Qwen3 全系（含 Qwen3-Next/Coder/VL）、Llama 3.1/3.3/4、
  DeepSeek-V3/V3.1/V3.2/V4-Flash/V4-Pro/R1、Kimi-K2/-K2-Thinking/-K3、
  GLM-4.5-Air/4.6/5、Hunyuan-A13B、Baichuan-M2、Seed-OSS、GPT-OSS、Mixtral、
  MiniCPM4、QwQ、Qwen2.5 …
- **20 款 GPU/NPU** — NVIDIA（H100/H200/H20/A100/B200/L40S/RTX 3090/4090/5090）、
  AMD MI300X、华为昇腾（910B/910C/950PR）、平头哥含光800、海光 Z100、昆仑芯 P800、
  寒武纪 MLU590、天数天垓150、沐曦 C500、摩尔线程 S5000

模型架构数据来自各模型公开的 `config.json`（ModelScope/HuggingFace）；硬件数据来自
官方规格表与公开规格汇总，每个条目在 `dsh/data.js` 中带有来源链接。官方未公开的
字段（如含光800 显存带宽、950 灵衢互联）标注 `~估算`，报告里会提示。

## 估算模型（可信度所在）

公式与系数全部公开，结论可复核：

- 权重 = 参数量 × 精度字节；KV = 2 × 层数 × KV头 × head_dim × 字节（GQA），
  MLA 为 `(kv_lora_rank + rope) × 层数 × 字节`，滑动窗口层按窗口折算（Qwen3-Next）
- 解码每 token ≈ max((权重+KV)/带宽, 计算) + TP allreduce（2·(TP−1)/TP × hidden ×
  batch × 2B，走 NVLink/HCCS/PCIe/节点上行）
- TTFT ≈ 2 × 激活参数量 × 输入 tokens / (算力 × TP × PP × MFU 0.5) + 通信
- 功耗 = TDP × 负载比例 + 22% 平台开销，PUE 1.25
- MoE 专家路由通信 ×1.35 修正；PP 每级气泡 10%

估算结果用真实部署锚点校验过（70B BF16 在 8×H100 ≈ 150 tok/s；DeepSeek-V3 FP8
放不进 8×80G、INT4 可以；Qwen3-8B INT4 单 4090 ≈ 170 tok/s），并扫描全部 760 个
模型×GPU 组合无异常。

国产 NPU 按规格表理想值估算，工具会提示 CANN/XPU/MUSA 软件栈实际利用率有差异，
正式采购前请用目标框架实测。

**开发、测试与发布流程（维护者）见 [docs/MAINTAINING.md](./docs/MAINTAINING.md)。**

## License

MIT
