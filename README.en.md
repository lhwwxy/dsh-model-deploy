# dsh-model-deploy

> 🌐 Language / 语言切换: **[English](./README.en.md)** · **[简体中文](./README.md)**

DeepSeek Harness (DSH) plugin for **LLM model selection & deployment analysis**.

Give it a model, a GPU/NPU, node count, precision, context length, concurrency and interconnect bandwidth — it estimates, with a transparent first-order analytical model:

- ✅/⚠️/❌ **deployability** (VRAM fit, precision support, interconnect checks) with fixes
- **memory**: per-GPU and cluster breakdown (weights / KV cache / activations / runtime)
- **latency**: time-to-first-token and end-to-end latency (idle & loaded)
- **throughput**: per-request and aggregate tokens/s, req/s, prefill tokens/s
- **power**: idle / typical / peak (PUE-adjusted), energy per token
- automatic **TP × PP × DP** strategy selection with reasoning

Built against the DeepSeek Harness architecture docs (`docs/architecture.md`): a model-facing
capability registered on `ctx.tools`, mounted as a bundle row in `cordis.patch.yml`.

## Install

```sh
dsh plugin --profile web add dsh-model-deploy
```

Or from a GitHub checkout:

```sh
dsh plugin --profile web add github:YOUR_ACCOUNT/dsh-model-deploy
```

> `--profile` is mandatory. The plugin registers two tools; restart the session
> (or the profile) after installing so the tool schemas join prompt assembly.

## Tools

### `model_deploy_analyze`

Main tool. Example call:

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

Fields: `model` (catalog id, or a custom-model JSON string), `gpu` (catalog id),
`gpusPerNode` (1–8), `nodes` (1–64), `precision` (`fp32|bf16|fp16|fp8|int8|int4|fp4`),
`ctx` / `batch` / `prompt` / `output` (tokens), `intraMode` (`auto|custom`) + `intraGBs`,
`interMode` (`none|ib400|ib800|roce100|roce200|custom`) + `interGBs`.

Returns a structured report (`status`, `issues`, `strategy`, `memory`, `performance`,
`power`, `assumptions`); the rendered output is a compact Chinese text report.

### `model_deploy_catalog`

Lists supported models (id, params, layers, context limit) and GPUs/NPUs
(id, memory, bandwidth, FP16/FP8/INT8/FP4, interconnect, TDP). Optional
`query` / `vendor` filters.

## Coverage

- **38 models** — Qwen3 family (incl. Qwen3-Next / Coder / VL), Llama 3.1/3.3/4,
  DeepSeek-V3/V3.1/V3.2/V4-Flash/V4-Pro/R1, Kimi-K2/-K2-Thinking/-K3, GLM-4.5-Air/4.6/5,
  Hunyuan-A13B, Baichuan-M2, Seed-OSS, GPT-OSS, Mixtral, MiniCPM4, QwQ, Qwen2.5 …
- **20 GPUs/NPUs** — NVIDIA (H100/H200/H20/A100/B200/L40S/RTX 3090/4090/5090),
  AMD MI300X, 华为昇腾 (910B/910C/950PR), 平头哥含光800, 海光 Z100, 昆仑芯 P800,
  寒武纪 MLU590, 天数天垓150, 沐曦 C500, 摩尔线程 S5000

Architecture data comes from each model's published `config.json`
(ModelScope/HuggingFace); hardware data from official datasheets and public
spec tables. Sources are linked per entry in `dsh/data.js`.

## Estimation model (the honest part)

All formulas and coefficients are public — that is what makes the numbers
auditable, not magic:

- weights = params × precision bytes; KV = 2 × layers × KV-heads × head_dim × bytes
  (GQA), or `(kv_lora_rank + rope)` × layers × bytes (MLA), with sliding-window
  layers prorated (Qwen3-Next)
- decode step ≈ max((weights + KV) / bandwidth, compute) + TP allreduce
  (2·(TP−1)/TP × hidden × batch × 2B over NVLink/HCCS/PCIe/node uplink)
- TTFT ≈ 2 × active params × prompt tokens / (FLOPs × TP × PP × MFU 0.5) + comm
- power = TDP × duty ratios + 22% platform overhead, PUE 1.25
- MoE expert all-to-all ×1.35 correction; PP bubble ×10%/stage

Verification: the engine is pinned by **22 regression tests** including
real-world anchors (70B BF16 on 8×H100 ≈ 150 tok/s; DeepSeek-V3 FP8 does not
fit 8×80GB, INT4 does; Qwen3-8B INT4 on one RTX 4090 ≈ 170 tok/s) plus a
NaN-free sweep across all 760 model×GPU combinations.

Non-CUDA NPUs are estimated at datasheet-ideal values; the tool flags that
CANN/XPU/MUSA software-stack maturity varies — benchmark before purchase.

## Development

```sh
npm test        # node --test (no dependencies)
```

`dsh/` is plain dependency-free ESM. Engine logic is shared with the
[companion website](https://github.com/YOUR_ACCOUNT/dsh-model-deploy/tree/main#companion-site)
(not required for the plugin).

## License

MIT
