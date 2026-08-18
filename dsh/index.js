// dsh-model-deploy —— DeepSeek Harness 插件（Host 侧）
//
// 注册两个模型可见工具：
//   model_deploy_catalog  查询支持的模型 / GPU-NPU 目录（id 查询）
//   model_deploy_analyze  模型选型部署分析：可部署性、显存、时延、吞吐、功耗
//
// 估算逻辑在 ./engine.js（纯函数，无依赖），数据集在 ./data.js。
// 架构依据 docs/architecture.md："Add a model-facing capability — register on
// ctx.tools"；注册由 tools 服务按插件生命周期自动回收（架构文档：
// registrations are effects that unwind when their plugin unloads）。
//
// 通过 cordis.patch.yml 挂载（package.json 的 dsh.bundle 声明入口）。

import { estimate } from './engine.js'
import { MODELS, GPUS, byId } from './data.js'

export const name = 'model-deploy'

export const inject = ['tools']

const PREC_LABEL = {
  fp32: 'FP32', bf16: 'BF16', fp16: 'FP16', fp8: 'FP8', int8: 'INT8', int4: 'INT4', fp4: 'FP4',
}

function fmt(v, d) {
  if (v === undefined || v === null || !isFinite(v)) return '—'
  d = d === undefined ? 1 : d
  return Number(v.toFixed(d)).toLocaleString('zh-CN')
}

// ---- 自定义模型：允许传入 JSON 字符串或对象 ----
function resolveModel(spec) {
  if (typeof spec === 'string') {
    const t = spec.trim()
    if (!t.startsWith('{')) return byId(MODELS, t)
    try {
      spec = JSON.parse(t)
    } catch {
      return null
    }
  }
  if (spec && typeof spec === 'object' && !Array.isArray(spec) && typeof spec.totalB === 'number') {
    const mla = spec.mla === true
    return {
      id: '__custom__',
      name: typeof spec.name === 'string' && spec.name.trim() ? spec.name.trim() : '自定义模型',
      org: '自定义',
      family: '自定义',
      dense: spec.activeB == null,
      totalB: spec.totalB,
      activeB: typeof spec.activeB === 'number' ? spec.activeB : null,
      layers: Math.max(1, Math.round(spec.layers || 32)),
      kvHeads: Math.max(1, Math.round(spec.kvHeads || 8)),
      headDim: Math.max(1, Math.round(spec.headDim || 128)),
      attnHeads: Math.max(1, Math.round(spec.attnHeads || 32)),
      hiddenSize: Math.max(1, Math.round(spec.hiddenSize || 4096)),
      maxCtx: Math.max(1, Math.round(spec.maxCtx || 32768)),
      mla,
      kvLoraRank: Math.max(1, Math.round(spec.kvLoraRank || 512)),
      ropeDim: Math.max(1, Math.round(spec.ropeDim || 64)),
      slideLayers: typeof spec.slideLayers === 'number' ? spec.slideLayers : undefined,
      slideWindow: typeof spec.slideWindow === 'number' ? spec.slideWindow : undefined,
      ctxNote: '用户自定义参数',
      src: null,
    }
  }
  return null
}

// ---- 目录工具 ----
function catalogTool() {
  return {
    name: 'model_deploy_catalog',
    description:
      '查询模型选型部署分析器支持的模型与 GPU/NPU 目录。先用它拿到准确的 model 与 gpu id（以及参数量/显存等关键规格），再调用 model_deploy_analyze 做部署分析。可选参数：query 按名称/厂商/系列过滤，vendor 按硬件厂商过滤。',
    parameters: {
      type: 'object',
      properties: {
        query: { type: 'string', description: '可选，按模型或硬件名称/厂商关键词过滤' },
        vendor: { type: 'string', description: '可选，按硬件厂商过滤，如 NVIDIA、华为昇腾、平头哥、昆仑芯、海光' },
      },
    },
    timeoutMs: 10_000,
    isConcurrencySafe: () => true,
    presentCall: (args) => ({ card: 'generic', title: '模型/硬件目录查询', kind: 'catalog', rawInput: args }),
    async execute(args) {
      const q = typeof args?.query === 'string' ? args.query.trim().toLowerCase() : ''
      const v = typeof args?.vendor === 'string' ? args.vendor.trim().toLowerCase() : ''
      const pick = (text) => !q || String(text).toLowerCase().includes(q)
      return {
        models: MODELS.filter((m) => pick(m.name + ' ' + m.org + ' ' + m.family + ' ' + m.id)).map((m) => ({
          id: m.id,
          name: m.name,
          org: m.org,
          family: m.family,
          totalB: m.totalB,
          activeB: m.activeB ?? null,
          layers: m.layers,
          maxCtx: m.maxCtx,
          mla: m.mla === true,
          ctxNote: m.ctxNote || '',
        })),
        gpus: GPUS.filter((g) => pick(g.name + ' ' + g.vendor + ' ' + g.id) && (!v || g.vendor.toLowerCase().includes(v))).map(
          (g) => ({
            id: g.id,
            name: g.name,
            vendor: g.vendor,
            memGB: g.memGB,
            memGBs: g.memGBs,
            fp16: g.fp16,
            fp8: g.fp8 || 0,
            int8: g.int8 || 0,
            fp4: g.fp4 || 0,
            link: g.link || (g.nvlinkGBs ? 'NVLink' : 'PCIe'),
            tdpW: g.tdpW,
            note: g.note || '',
          }),
        ),
      }
    },
  }
}

// ---- 分析工具 ----
function analyzeTool() {
  return {
    name: 'model_deploy_analyze',
    description:
      'LLM 模型选型部署分析：给定模型与 GPU/NPU 型号、每节点卡数、节点(机器)数、模型精度、上下文长度、并发数、输入/输出长度、卡间与节点间互联带宽，用一阶解析模型估算：能否部署(显存/精度/互联检查)、单卡与集群显存占用、首 token 时延(TTFT)、整体时延、解码与系统吞吐、功耗与每 token 能耗，并自动给出 TP/PP/DP 并行策略与工程建议。model 与 gpu 参数填 model_deploy_catalog 返回的 id；模型也支持传入自定义参数 JSON 字符串({totalB, activeB, layers, kvHeads, headDim, attnHeads, hiddenSize, maxCtx, mla...})。分析基于公开规格与经验系数(见 README 假设说明)，用于选型初筛，正式采购前需实测。',
    parameters: {
      type: 'object',
      properties: {
        model: { type: 'string', description: '模型 id（model_deploy_catalog 查询），或自定义模型参数 JSON 字符串' },
        gpu: { type: 'string', description: 'GPU/NPU id（model_deploy_catalog 查询），如 h100、h200、b200、ascend910b、ascend910c、rtx4090、kunlun-p800' },
        gpusPerNode: { type: 'integer', minimum: 1, maximum: 8, description: '每节点（每台机器）卡数' },
        nodes: { type: 'integer', minimum: 1, maximum: 64, description: '节点数（机器数量）' },
        precision: { type: 'string', enum: ['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4', 'fp4'], description: '模型精度' },
        ctx: { type: 'integer', minimum: 1, description: '上下文长度(tokens)，KV 缓存按此预留' },
        batch: { type: 'integer', minimum: 1, description: '并发数（同时处理的序列数）' },
        prompt: { type: 'integer', minimum: 1, description: '输入长度(tokens)，用于首 token 时延计算' },
        output: { type: 'integer', minimum: 1, description: '输出长度(tokens)，用于整体时延计算' },
        intraMode: { type: 'string', enum: ['auto', 'custom'], description: '卡间互联：auto 按硬件规格，custom 配合 intraGBs' },
        intraGBs: { type: 'number', minimum: 0, description: '自定义卡间带宽 GB/s（intraMode=custom 时生效）' },
        interMode: { type: 'string', enum: ['none', 'ib400', 'ib800', 'roce100', 'roce200', 'custom'], description: '节点间互联：none 单机；ib400/ib800=InfiniBand 400G/800G×8；roce100/200；custom 配合 interGBs' },
        interGBs: { type: 'number', minimum: 0, description: '自定义节点上行带宽 GB/s（interMode=custom 时生效）' },
      },
      required: ['model', 'gpu'],
    },
    timeoutMs: 15_000,
    isConcurrencySafe: () => true,
    presentCall: (args) => ({ card: 'generic', title: '模型选型部署分析', kind: 'analysis', rawInput: args }),
    async execute(args, exec) {
      const model = resolveModel(args.model)
      const gpu = byId(GPUS, typeof args.gpu === 'string' ? args.gpu.trim() : '')
      if (!model) {
        throw new Error(
          `未知模型 "${args.model}"。先用 model_deploy_catalog 查询支持的模型 id；或传入自定义模型参数 JSON（至少含 totalB、layers）。`,
        )
      }
      if (!gpu) {
        throw new Error(`未知硬件 "${args.gpu}"。先用 model_deploy_catalog 查询支持的 GPU/NPU id。`)
      }
      if (args.precision && !['fp32', 'bf16', 'fp16', 'fp8', 'int8', 'int4', 'fp4'].includes(args.precision)) {
        throw new Error(`不支持的精度 "${args.precision}"，可选：fp32/bf16/fp16/fp8/int8/int4/fp4`)
      }
      const report = estimate({
        model,
        gpu,
        gpusPerNode: args.gpusPerNode ?? 8,
        nodes: args.nodes ?? 1,
        precision: args.precision || 'bf16',
        ctx: args.ctx ?? 32768,
        batch: args.batch ?? 8,
        prompt: args.prompt ?? 1024,
        output: args.output ?? 512,
        intraMode: args.intraMode || 'auto',
        intraGBs: args.intraGBs || 0,
        interMode: args.interMode || 'none',
        interGBs: args.interGBs || 0,
      })
      return {
        model: model.name,
        gpu: gpu.name,
        totalGPUs: report.strategy.totalGPUs,
        precision: PREC_LABEL[args.precision || 'bf16'] || args.precision,
        ctx: args.ctx ?? 32768,
        batch: args.batch ?? 8,
        status: report.status,
        issues: report.issues,
        strategy: report.strategy,
        memory: report.mem,
        performance: report.perf,
        power: report.power,
        assumptions: {
          hbmUtil: report.knobs.hbmUtilNvlink,
          prefillMFU: report.knobs.prefillMFU,
          pue: report.knobs.pue,
          note: '一阶解析估算（权重=参数量×精度字节；KV=2×层数×KV头×head_dim×字节，MLA 按压缩公式；解码受内存带宽约束；TP allreduce 计入互联带宽）。国产 NPU 为规格表理想值，实际依赖 CANN/XPU/MUSA 适配成熟度。选型初筛用，正式决策需基准测试。',
        },
      }
    },
  }
}

// ---- 结果渲染：把结构化报告转成给用户看的文本 ----
export function formatReport(r) {
  if (!r || typeof r !== 'object') return String(r)
  const s = r.strategy || {}
  const m = r.memory || {}
  const p = r.performance || {}
  const w = r.power || {}
  const titles = { ok: '✅ 可部署', tight: '⚠️ 可部署但显存紧张', fail: '❌ 不可部署' }
  const lines = []
  lines.push(`# 模型选型部署分析：${r.model} · ${r.precision} · ${r.gpu} ×${r.totalGPUs}`)
  lines.push('')
  lines.push(`**结论：${titles[r.status] || r.status}**`)
  if (s.tp) lines.push(`并行策略：TP${s.tp} × PP${s.pp} × DP${s.dp}（每副本 ${s.perReplicaGPUs} 卡 × ${s.dp} 副本）`)
  if (m.perGPU) {
    lines.push(
      `显存（单卡）：${fmt(m.perGPU.total)} / ${fmt(m.perGPU.usable)} GB（利用率 ${fmt(m.perGPU.util * 100, 0)}%）` +
        ` · 权重 ${fmt(m.perGPU.weights)} GB + KV ${fmt(m.perGPU.kv)} GB + 激活/运行时 ${fmt(m.perGPU.activ + m.perGPU.runtime)} GB`,
    )
  }
  lines.push(
    `时延：TTFT ${fmt(p.ttftIdleMs, 0)} ms（满载约 ${fmt(p.ttftLoadedMs, 0)} ms）· 整体 ${fmt(p.totalIdleMs, 0)} ms（${fmt(argsOut(r))}）`,
  )
  lines.push(
    `吞吐：单请求 ${fmt(p.perReqTPSIdle, 0)} tok/s（当前并发 ${fmt(p.perReqTPS, 0)}）· 系统 ${fmt(p.aggTPS, 0)} tok/s ≈ ${fmt(p.rps, 2)} req/s`,
  )
  lines.push(
    `功耗：空闲 ${fmt(w.idleW / 1000, 2)} kW · 典型 ${fmt(w.avgW / 1000, 2)} kW · 峰值 ${fmt(w.peakW / 1000, 2)} kW（PUE 后 ${fmt(w.peakPueW / 1000, 2)}）· 每 token ${fmt(w.jPerToken, 1)} J`,
  )
  const errs = (r.issues || []).filter((i) => i.level === 'error')
  const warns = (r.issues || []).filter((i) => i.level === 'warn')
  if (errs.length) lines.push('', `**问题：** ${errs.map((i) => i.text + (i.sug ? '（建议：' + i.sug + '）' : '')).join('；')}`)
  if (warns.length) lines.push('', `**注意：** ${warns.map((i) => i.text).join('；')}`)
  return lines.join('\n')
}

function argsOut(r) {
  // formatReport 从 report 对象推断输入/输出长度：保存在 memory/performance 之外，这里用默认说明
  return '输入长度×输出长度按调用参数计'
}

export function apply(ctx, config = {}) {
  if (config.registerCatalog !== false) {
    try {
      ctx.tools.register(catalogTool())
    } catch (error) {
      console.error('[dsh-model-deploy] model_deploy_catalog 注册失败:', error)
    }
  }
  if (config.registerAnalyze !== false) {
    try {
      ctx.tools.register(analyzeTool())
    } catch (error) {
      console.error('[dsh-model-deploy] model_deploy_analyze 注册失败:', error)
    }
  }
}
