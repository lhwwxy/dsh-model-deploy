# 维护者手册

> 读者：本仓库的维护者/贡献者。安装使用者请看根目录 README。

## 开发与测试

```sh
npm test        # node --test，零依赖
```

- `dsh/` 为无运行时依赖的纯 ESM（`index.js` 插件入口、`engine.js` 估算引擎、`data.js` 数据集）
- 估算引擎与配套选型网站（`model-selection-site/`，本地未开源）共用同一实现；改引擎后两边需同步
- 22 项回归测试包含真实部署锚点（70B 8×H100 ≈150 tok/s 区间、V3 FP8 放不进 8×80G 等）与 760 组合无 NaN 扫描，改引擎必须全绿

## 如何添加模型 / GPU

编辑 `dsh/data.js`，向 `MODELS` / `GPUS` 数组追加条目：

```js
// 模型（稠密）
{ id: 'my-model', name: 'MyModel', org: '厂商', family: '系列', dense: true,
  totalB: 7, activeB: null, layers: 32, kvHeads: 8, headDim: 128,
  attnHeads: 32, hiddenSize: 4096, maxCtx: 32768, ctxNote: '官方 32K',
  src: 'https://modelscope.cn/models/.../resolve/master/config.json' }

// MoE 模型：dense:false + activeB；MLA 模型：mla:true + kvLoraRank + ropeDim
// 混合注意力（滑动窗口）：加 slideLayers / slideWindow

// GPU/NPU
{ id: 'my-gpu', name: 'My GPU', vendor: '厂商', gen: '代际',
  memGB: 80, memGBs: 3350, fp16: 989, fp8: 1979, int8: 1979, fp4: 0,
  fp4Native: false, link: 'NVLink', nvlinkGBs: 900, pcieGen: 5, tdpW: 700,
  datacenter: true, ecosystem: 'npu', note: '来源与置信标注',
  src: 'https://官方规格页' }
```

字段含义见 `dsh/data.js` 顶部注释；官方未公开的字段标注 `~估算` 并写进 `note`。

## 发布流程

1. **npm**：`npm publish --otp=XXXXXX`（账号开了 2FA；或用 Automation 类型令牌免验证码）
2. **GitHub**：main 推送到 [lhwwxy/dsh-model-deploy](https://github.com/lhwwxy/dsh-model-deploy)
3. **目录站**：仓库 topic 已含 `dsh-plugin`（周期同步自动收录）；也可到 <https://dshmarketplace.dev/submit> 提交（人工审核）
4. （可选）发到 LINUX DO 社区可获得 verified 徽章

> 注意：npm 页面显示的 README 是发布时快照，文档改动需发新版本才会生效。
> 建议按 SemVer 用 `npm version patch` 打 tag；重大文档结构调整可补发 patch 版本。
