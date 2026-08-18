// 插件加载冒烟测试：用 mock ctx 验证 apply() 注册两个工具且不抛错
import test from 'node:test'
import assert from 'node:assert/strict'
import { apply, name, inject, formatReport } from '../dsh/index.js'

test('插件声明：inject 包含 tools', () => {
  assert.equal(name, 'model-deploy')
  assert.ok(inject.includes('tools'))
})

test('apply(ctx) 注册两个工具', () => {
  const registered = []
  const ctx = {
    tools: {
      register(tool) {
        registered.push(tool)
        return () => {}
      },
    },
  }
  apply(ctx)
  assert.equal(registered.length, 2)
  const names = registered.map((t) => t.name).sort()
  assert.deepEqual(names, ['model_deploy_analyze', 'model_deploy_catalog'])
  for (const t of registered) {
    assert.equal(typeof t.execute, 'function')
    assert.equal(t.parameters.type, 'object')
  }
})

test('analyze 工具对未知 id 给出可读错误', async () => {
  const tools = {}
  apply({ tools: { register: (t) => (tools[t.name] = t) } })
  await assert.rejects(
    () => tools.model_deploy_analyze.execute({ model: 'no-such-model', gpu: 'h100' }, {}),
    /model_deploy_catalog/,
  )
  await assert.rejects(
    () => tools.model_deploy_analyze.execute({ model: 'qwen3-8b', gpu: 'no-such-gpu' }, {}),
    /model_deploy_catalog/,
  )
})

test('analyze 工具端到端：V3 INT4 · 8×昇腾910B', async () => {
  const tools = {}
  apply({ tools: { register: (t) => (tools[t.name] = t) } })
  const r = await tools.model_deploy_analyze.execute(
    { model: 'deepseek-v3', gpu: 'ascend910b', gpusPerNode: 8, nodes: 1, precision: 'int4', ctx: 131072, batch: 4 },
    {},
  )
  assert.equal(r.status, 'ok')
  assert.equal(r.totalGPUs, 8)
  assert.ok(r.performance.perReqTPSIdle > 8 && r.performance.perReqTPSIdle < 80)
  const text = formatReport(r)
  assert.ok(text.includes('可部署'))
  assert.ok(text.includes('TP8'))
})

test('catalog 工具返回模型与硬件目录', async () => {
  const tools = {}
  apply({ tools: { register: (t) => (tools[t.name] = t) } })
  const c = await tools.model_deploy_catalog.execute({ query: '昇腾' }, {})
  assert.ok(c.gpus.length >= 3)
  assert.ok(c.gpus.every((g) => g.name.includes('昇腾')))
  const all = await tools.model_deploy_catalog.execute({}, {})
  assert.ok(all.models.length >= 38)
  assert.ok(all.gpus.length >= 20)
})
