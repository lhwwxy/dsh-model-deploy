// 引擎基准锚点校验（对照公开部署实践/基准的合理区间；与配套网站同源）
import test from 'node:test'
import assert from 'node:assert/strict'
import * as D from '../dsh/data.js'
import * as E from '../dsh/engine.js'

function base(modelId, gpuId, opts) {
  const o = opts || {}
  return {
    model: D.byId(D.MODELS, modelId),
    gpu: D.byId(D.GPUS, gpuId),
    gpusPerNode: o.gpusPerNode || 8,
    nodes: o.nodes || 1,
    precision: o.precision || 'bf16',
    ctx: o.ctx || 32768,
    batch: o.batch || 1,
    prompt: o.prompt || 1024,
    output: o.output || 512,
    intraMode: o.intraMode || 'auto',
    intraGBs: 0,
    interMode: o.interMode || 'none',
    interGBs: 0,
  }
}
function between(t, v, lo, hi) {
  assert.ok(v >= lo && v <= hi, `${t}: v=${v} 期望 [${lo}, ${hi}]`)
}

test('KV 公式：DeepSeek-V3 MLA = 70,272 B/token (BF16)', () => {
  assert.equal(E.kvPerTokenBytes(D.byId(D.MODELS, 'deepseek-v3'), 2), 70272)
})
test('KV 公式：Llama-70B = 327,680 B/token (BF16)', () => {
  assert.equal(E.kvPerTokenBytes(D.byId(D.MODELS, 'llama33-70b'), 2), 327680)
})
test('KV 公式：Qwen3-Next 滑动窗口 (256K ctx) = 40,960 B', () => {
  assert.equal(E.kvPerTokenBytes(D.byId(D.MODELS, 'qwen3-next'), 2, 262144), 40960)
})

test('70B BF16 · 8×H100', () => {
  const r = E.estimate(base('llama33-70b', 'h100', {}))
  assert.equal(r.status, 'ok')
  between('解码速度', r.perf.perReqTPSIdle, 100, 260)
  between('TTFT', r.perf.ttftIdleMs, 15, 150)
  between('单卡显存', r.mem.perGPU.total, 15, 32)
})
test('70B BF16 · 1×H100 → 不可部署且给加卡建议', () => {
  const r = E.estimate(base('llama33-70b', 'h100', { gpusPerNode: 1, nodes: 1 }))
  assert.equal(r.status, 'fail')
  assert.ok(r.issues.some((i) => i.level === 'error' && i.sug.includes('卡')))
})
test('70B BF16 · 2×H100 · 32K 紧张 / 64K 放不下', () => {
  assert.equal(E.estimate(base('llama33-70b', 'h100', { gpusPerNode: 2, nodes: 1, ctx: 32768 })).status, 'tight')
  assert.equal(E.estimate(base('llama33-70b', 'h100', { gpusPerNode: 2, nodes: 1, ctx: 65536 })).status, 'fail')
})

test('DeepSeek-V3：FP8 放不进 8×H100，INT4 可以', () => {
  assert.equal(E.estimate(base('deepseek-v3', 'h100', { precision: 'fp8' })).status, 'fail')
  const r = E.estimate(base('deepseek-v3', 'h100', { precision: 'int4' }))
  assert.equal(r.status, 'ok')
  between('解码速度', r.perf.perReqTPSIdle, 25, 95)
})

test('Qwen3-8B INT4 · 单 4090', () => {
  const r = E.estimate(base('qwen3-8b', 'rtx4090', { gpusPerNode: 1, nodes: 1, precision: 'int4', ctx: 8192 }))
  assert.equal(r.status, 'ok')
  between('解码速度', r.perf.perReqTPSIdle, 90, 260)
})

test('Llama-4 Maverick：BF16 8 卡放不下，FP8 可以', () => {
  assert.equal(E.estimate(base('llama4-maverick', 'h100', {})).status, 'fail')
  const r = E.estimate(base('llama4-maverick', 'h100', { precision: 'fp8' }))
  assert.equal(r.status, 'ok')
  between('解码速度', r.perf.perReqTPSIdle, 25, 100)
})

test('Qwen3-235B FP8 · 8×H100', () => {
  const r = E.estimate(base('qwen3-235b-a22b', 'h100', { precision: 'fp8' }))
  assert.equal(r.status, 'ok')
  between('解码速度', r.perf.perReqTPSIdle, 50, 150)
})

test('A100 + FP8 → 报错', () => {
  const r = E.estimate(base('qwen3-8b', 'a100', { gpusPerNode: 1, nodes: 1, precision: 'fp8' }))
  assert.equal(r.status, 'fail')
  assert.ok(r.issues.some((i) => i.level === 'error' && i.text.includes('FP8')))
})

test('Kimi-K2 FP8 · 16×H200（2 节点 IB400）', () => {
  const r = E.estimate(base('kimi-k2', 'h200', { gpusPerNode: 8, nodes: 2, precision: 'fp8', ctx: 131072, batch: 8, interMode: 'ib400' }))
  assert.equal(r.status, 'ok')
  between('每请求解码速度', r.perf.perReqTPS, 8, 70)
})

test('Qwen3-32B INT4 · 2×4090（PCIe TP）', () => {
  const r = E.estimate(base('qwen3-32b', 'rtx4090', { gpusPerNode: 2, nodes: 1, precision: 'int4', ctx: 32768 }))
  assert.ok(r.status === 'ok' || r.status === 'tight')
  between('解码速度', r.perf.perReqTPSIdle, 25, 90)
})

test('功耗锚点：8×H100', () => {
  const r = E.estimate(base('llama33-70b', 'h100', {}))
  between('典型功耗 kW', r.power.avgW / 1000, 3.5, 6)
  between('每 token 能耗 J', r.power.jPerToken, 10, 80)
})

test('新模型锚点', () => {
  assert.equal(E.estimate(base('deepseek-v3.1', 'h100', { precision: 'fp8' })).status, 'fail')
  const g = E.estimate(base('glm-4.6', 'h200', { precision: 'int4', ctx: 131072 }))
  assert.equal(g.status, 'ok')
  between('GLM-4.6 解码速度', g.perf.perReqTPSIdle, 50, 220)
  const k = E.estimate(base('kimi-k3', 'h200', { gpusPerNode: 8, nodes: 2, precision: 'int4', ctx: 131072, interMode: 'ib400' }))
  assert.equal(k.status, 'ok')
  between('Kimi-K3 解码速度', k.perf.perReqTPSIdle, 10, 90)
  assert.equal(E.estimate(base('deepseek-v4-flash', 'h200', { precision: 'fp8', ctx: 131072 })).status, 'ok')
})

test('国产 NPU 锚点', () => {
  const b = E.estimate(base('deepseek-v3', 'ascend910b', { gpusPerNode: 8, precision: 'int4', ctx: 131072 }))
  assert.equal(b.status, 'ok')
  between('V3 INT4 · 8×910B 解码速度', b.perf.perReqTPSIdle, 8, 80)
  const c = E.estimate(base('deepseek-v4-flash', 'ascend910c', { gpusPerNode: 8, precision: 'fp8', ctx: 131072 }))
  assert.equal(c.status, 'ok')
  const h = E.estimate(base('qwen3-8b', 'hanguang800', { gpusPerNode: 1, precision: 'bf16', ctx: 8192 }))
  assert.equal(h.status, 'fail')
  assert.ok(h.issues.some((i) => i.level === 'error'))
  const m = E.estimate(base('minicpm4', 'hanguang800', { gpusPerNode: 1, precision: 'int8', ctx: 8192 }))
  assert.equal(m.status, 'ok')
})

test('全模型×全GPU 网格无 NaN/负值（INT8）', () => {
  let bad = 0
  for (const m of D.MODELS) {
    for (const g of D.GPUS) {
      const r = E.estimate(base(m.id, g.id, { precision: 'int8', ctx: 32768, batch: 4 }))
      for (const v of [r.mem.perGPU.total, r.perf.ttftIdleMs, r.perf.stepMs, r.perf.aggTPS, r.power.avgW, r.power.jPerToken]) {
        if (!isFinite(v) || v < 0) bad++
      }
    }
  }
  assert.equal(bad, 0, `bad=${bad} / ${D.MODELS.length * D.GPUS.length} 组合`)
})
