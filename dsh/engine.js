// 一阶解析估算引擎（纯函数，无运行时依赖；与配套网站 model-selection-site 同一实现，已通过 41 项基准锚点校验）


  var GiB = 1073741824;

  var PREC = {
    fp32: { bytes: 4 }, bf16: { bytes: 2 }, fp8: { bytes: 1 },
    int8: { bytes: 1 }, int4: { bytes: 0.5 }, fp4: { bytes: 0.5 }
  };

  var K = {
    hbmUtilNvlink: 0.90,   // 数据中心 NVLink 平台 HBM 有效带宽利用率(解码)
    hbmUtilPcie: 0.75,     // 消费卡/无 NVLink 平台
    memUsableRatio: 0.97,  // 单卡可用显存比例(驱动/ECC 保留)
    runtimeGB: 2.5,        // 每卡固定运行时开销(CUDA context/框架)
    activRatio: 0.02,      // 激活内存占权重的比例
    prefillMFU: 0.50,      // 预填充阶段算力利用率(MFU)
    decodeComputeUtil: 0.60,
    allreduceEff: 0.85,    // allreduce 有效带宽系数
    allreduceEffPcie: 0.60,
    latNvlinkUs: 2,        // 单次 allreduce 启动延迟
    latPcieUs: 8,
    moeCommFactor: 1.35,   // MoE 专家路由 all-to-all 通信放大(粗修正)
    ppBubblePerStage: 0.10,// PP 每级流水气泡系数
    platformPowerRatio: 0.22, // 平台开销(CPU/内存/电源) 占 GPU TDP 比例
    decodePowerRatio: 0.62,   // 解码阶段 GPU 平均功耗占 TDP
    idlePowerRatio: 0.30,     // 空闲 GPU 功耗占 TDP
    pue: 1.25
  };

  var INTER = { ib400: 400, ib800: 800, roce100: 100, roce200: 200 };

  function divisors(n) {
    var r = [];
    for (var i = 1; i * i <= n; i++) {
      if (n % i === 0) { r.push(i); var j = n / i; if (j !== i) r.push(j); }
    }
    return r.sort(function (a, b) { return a - b; });
  }

  function tflopsFor(p, gpu) {
    switch (p) {
      case 'fp32': return gpu.fp16 * 0.5;                    // 按 TF32 Tensor Core 路径
      case 'bf16': case 'fp16': return gpu.fp16;
      case 'fp8': return gpu.fp8 || 0;
      case 'int8': return gpu.int8 || gpu.fp16 * 2;
      case 'int4': return gpu.fp4Native ? gpu.fp4 * 0.9 : (gpu.fp16 || gpu.int8 / 2) * 0.9; // 反量化路径
      case 'fp4': return gpu.fp4Native ? gpu.fp4 * 0.9 : (gpu.fp16 || gpu.int8 / 2) * 0.9;
    }
    return gpu.fp16;
  }

  // 单 token 的 KV 缓存字节数（K+V，按所选 KV 元素精度）
  // ctx: 上下文长度；混合注意力模型（部分层滑动窗口）按窗口比例折算
  function kvPerTokenBytes(m, elemBytes, ctx) {
    if (m.mla) return (m.kvLoraRank + m.ropeDim) * m.layers * elemBytes;
    var slide = m.slideLayers || 0;
    var full = m.layers - slide;
    var ratio = 1;
    if (slide && ctx) ratio = Math.min(ctx, m.slideWindow || ctx) / ctx;
    return 2 * (full + slide * ratio) * m.kvHeads * m.headDim * elemBytes;
  }

  function ppBubble(pp) { return 1 + K.ppBubblePerStage * (pp - 1); }

  function memPerGPU(wpg, kvseq, bpr) {
    return wpg + kvseq * bpr + wpg * K.activRatio + K.runtimeGB;
  }

  // 放得下该模型所需的最小卡数（batch=1, 当前精度/上下文）
  // power2=true 时仅考虑常规配置（TP/PP ∈ {1,2,4,8}）
  function minCards(m, gpu, prec, ctx, power2) {
    var weightGB = m.totalB * 1e9 * PREC[prec].bytes / GiB;
    var elemBytes = (prec === 'fp8' || prec === 'int8' || prec === 'int4' || prec === 'fp4') ? (gpu.fp8 ? 1 : 2) : 2;
    var kvb = kvPerTokenBytes(m, elemBytes, ctx);
    var usable = gpu.memGB * K.memUsableRatio;
    var okDims = function (v) { return !power2 || v === 1 || v === 2 || v === 4 || v === 8; };
    for (var n = 1; n <= 512; n++) {
      var tps = divisors(n);
      for (var i = tps.length - 1; i >= 0; i--) {
        var tp = tps[i];
        if (tp > Math.min(8, m.attnHeads) || !okDims(tp)) continue;
        var pps = divisors(n / tp);
        for (var j = 0; j < pps.length; j++) {
          var pp = pps[j];
          if (pp > m.layers || !okDims(pp)) continue;
          var wpg = weightGB / (tp * pp);
          var kvseq = kvb * ctx / GiB / (tp * pp);
          if (memPerGPU(wpg, kvseq, 1) <= usable) return { n: n, tp: tp, pp: pp };
        }
      }
    }
    return { n: 0, tp: 0, pp: 0 };
  }

  function estimate(inp) {
    var m = inp.model, gpu = inp.gpu;
    var totalGPUs = Math.max(1, Math.round(inp.gpusPerNode) * Math.round(inp.nodes));
    var p = inp.precision;
    var ctx = Math.max(1, Math.round(inp.ctx || 4096));
    var batch = Math.max(1, Math.round(inp.batch || 1));
    var prompt = Math.max(1, Math.round(inp.prompt || 1024));
    var output = Math.max(1, Math.round(inp.output || 512));
    var issues = [];

    // ---- 基础量 ----
    var weightGB = m.totalB * 1e9 * PREC[p].bytes / GiB;
    var kvElemBytes = (p === 'fp8' || p === 'int8' || p === 'int4' || p === 'fp4') ? (gpu.fp8 ? 1 : 2) : 2;
    var kvPerTokenB = kvPerTokenBytes(m, kvElemBytes, ctx);
    var activeB = m.activeB || m.totalB;
    var tf = tflopsFor(p, gpu);

    // ---- 精度 / 上下文检查 ----
    if (p === 'fp8' && !gpu.fp8) issues.push({ level: 'error', text: gpu.name + ' 无原生 FP8 支持（需 Hopper/Ada/Blackwell 或 MI300X）', sug: '改用 BF16/FP16 或 INT8' });
    if (p === 'fp4' && !gpu.fp4Native) issues.push({ level: 'warn', text: 'FP4 仅在 Blackwell/昇腾950 原生支持，当前按反量化路径估算（性能按 FP16 计）', sug: '非 Blackwell 平台建议 INT4(AWQ/GPTQ) 或 FP8' });
    if (p === 'fp32') issues.push({ level: 'warn', text: 'FP32 极少用于推理；算力按 TF32 Tensor Core 路径估算', sug: '推荐 BF16/FP16' });
    if (tf <= 0) issues.push({ level: 'error', text: gpu.name + ' 无 ' + p.toUpperCase() + ' 算力（整型专用 NPU 等仅支持 INT8/INT4）', sug: '改用 INT8 或 INT4 精度' });
    if (gpu.ecosystem === 'npu') issues.push({ level: 'info', text: '非 CUDA 生态硬件（' + (gpu.vendor || '国产 NPU') + '）：实际吞吐取决于 CANN/XPU/MUSA 等框架适配成熟度，本估算基于规格表理想值', sug: '国产 NPU 建议参考 vLLM-Ascend、MindIE 等已适配框架的实测数据' });
    if (ctx > m.maxCtx) issues.push({ level: 'warn', text: '上下文 ' + ctx.toLocaleString() + ' 超出官方上限 ' + m.maxCtx.toLocaleString(), sug: '超出部分需 YaRN/iRoPE 外推，质量与显存需实测' });

    // ---- 互联带宽 ----
    var intraGBs;
    if (inp.intraMode === 'custom') intraGBs = Math.max(0, Number(inp.intraGBs) || 0);
    else intraGBs = gpu.nvlinkGBs || (gpu.pcieGen >= 5 ? 50 : 25);

    var interGBs = 0;
    if (inp.nodes > 1 && inp.interMode && inp.interMode !== 'none') {
      interGBs = inp.interMode === 'custom' ? Math.max(0, Number(inp.interGBs) || 0) : (INTER[inp.interMode] || 0);
    }
    var interEnabled = interGBs > 0;
    if (inp.nodes > 1 && !interEnabled && totalGPUs > inp.gpusPerNode) {
      issues.push({ level: 'info', text: '多节点但未配置节点间高速互联，张量并行(TP) 将被限制在单节点内', sug: '跨节点扩展需 IB/RoCE；否则只能数据并行(DP)/流水并行(PP)' });
    }

    // ---- 并行策略搜索（TP×PP×DP） ----
    // 偏好(贴近生产实践): 1) 放得下且余量≥8%  2) TP 留在节点内(NVLink)
    // 3) TP 尽量大(解码是带宽瓶颈, TP 分摊权重读取)  4) PP 尽量小  5) DP 尽量大
    var usable = gpu.memGB * K.memUsableRatio;
    var maxTP = Math.min(16, m.attnHeads || 1, interEnabled ? totalGPUs : Math.min(inp.gpusPerNode, totalGPUs));
    var cands = [];
    divisors(totalGPUs).forEach(function (tp) {
      if (tp > maxTP) return;
      var rem = totalGPUs / tp;
      divisors(rem).forEach(function (pp) {
        if (pp > m.layers) return;
        var dp = rem / pp;
        var wpg = weightGB / (tp * pp);
        var kvseq = kvPerTokenB * ctx / GiB / (tp * pp);
        var bpr = Math.ceil(batch / dp);
        var kvpg = kvseq * bpr;
        var mempg = memPerGPU(wpg, kvseq, bpr);
        var fits = mempg <= usable;
        cands.push({ tp: tp, pp: pp, dp: dp, wpg: wpg, kvseq: kvseq, bpr: bpr, kvpg: kvpg, mempg: mempg, fits: fits, util: mempg / usable, crossNode: tp > inp.gpusPerNode });
      });
    });
    function betterThan(a, b) {
      if ((a.crossNode ? 1 : 0) !== (b.crossNode ? 1 : 0)) return (a.crossNode ? 1 : 0) < (b.crossNode ? 1 : 0);
      if (a.tp !== b.tp) return a.tp > b.tp;
      if (a.pp !== b.pp) return a.pp < b.pp;
      return a.dp > b.dp;
    }
    var pool = cands.filter(function (c) { return c.fits && c.util <= 0.92; });
    if (!pool.length) pool = cands.filter(function (c) { return c.fits; });
    var best;
    if (pool.length) {
      best = pool[0];
      for (var bi = 1; bi < pool.length; bi++) if (betterThan(pool[bi], best)) best = pool[bi];
    } else {
      best = cands[0]; // 全放不下: 取显存占用最小者用于展示与建议
      for (var bj = 1; bj < cands.length; bj++) {
        if (cands[bj].mempg < best.mempg - 1e-9 ||
            (Math.abs(cands[bj].mempg - best.mempg) < 1e-9 && betterThan(cands[bj], best))) best = cands[bj];
      }
    }

    var tp = best.tp, pp = best.pp, dp = best.dp, bpr = best.bpr;
    var hbmUtil = gpu.nvlinkGBs ? K.hbmUtilNvlink : K.hbmUtilPcie;

    // ---- 通信带宽（TP allreduce） ----
    var commGBs = 0;
    if (tp > 1) {
      commGBs = intraGBs;
      if (tp > inp.gpusPerNode) commGBs = Math.min(intraGBs, interGBs / Math.max(1, inp.gpusPerNode));
    }
    commGBs = Math.max(commGBs, 1e-9);
    var allreduceEff = gpu.nvlinkGBs ? K.allreduceEff : K.allreduceEffPcie;
    var latUs = gpu.nvlinkGBs ? K.latNvlinkUs : K.latPcieUs;
    var moeF = m.activeB ? K.moeCommFactor : 1;

    if (tp > 1 && !gpu.nvlinkGBs) issues.push({ level: 'warn', text: '该卡无高速卡间互联（NVLink/HCCS 未配置），TP=' + tp + ' 的通信按 PCIe（有效带宽约 ' + intraGBs + ' GB/s）估算，性能损失显著', sug: '建议模型放单卡(TP=1)；多卡 TP 仅适合带宽需求低的场景，或选用带高速互联的机型（如 Atlas 800I A2 的 HCCS）' });
    if (best.crossNode) issues.push({ level: 'warn', text: 'TP=' + tp + ' 跨节点，allreduce 带宽受节点上行 ' + interGBs + ' GB/s 限制', sug: '跨节点 TP 建议 ≥400G InfiniBand；或增大单节点卡数/单卡显存' });

    function commPerStepFor(bprLocal) {
      if (tp <= 1) return 0;
      var bytes = 2 * (tp - 1) / tp * m.hiddenSize * bprLocal * 2;
      return m.layers * (bytes / (commGBs * 1e9 * allreduceEff) + latUs * 1e-6) * moeF;
    }

    // ---- 解码每 token 时间 ----
    var memTime = (best.wpg + best.kvpg) / (gpu.memGBs * hbmUtil);
    var computeTime = (2 * activeB * 1e9 * bpr / (tp * pp)) / (tflopsFor(p, gpu) * 1e12 * K.decodeComputeUtil);
    var commMs = commPerStepFor(bpr) * 1000;
    var step = Math.max(memTime, computeTime) + commPerStepFor(bpr);

    var mtB1 = (best.wpg + best.kvseq) / (gpu.memGBs * hbmUtil);
    var ctB1 = (2 * activeB * 1e9 / (tp * pp)) / (tflopsFor(p, gpu) * 1e12 * K.decodeComputeUtil);
    var stepB1 = Math.max(mtB1, ctB1) + commPerStepFor(1);

    // ---- 预填充 / 首 token ----
    var tflopsTot = tflopsFor(p, gpu) * 1e12 * tp * pp * K.prefillMFU;
    function commPrefillFor(tokens) {
      if (tp <= 1) return 0;
      var bytes = 2 * (tp - 1) / tp * m.hiddenSize * tokens * 2;
      return m.layers * (bytes / (commGBs * 1e9 * allreduceEff) + latUs * 1e-6) * moeF;
    }
    function ttftFor(tokens) {
      var comp = 2 * activeB * 1e9 * tokens / tflopsTot;
      var comm = commPrefillFor(tokens);
      return Math.max(comp, comm * 0.5) * ppBubble(pp);
    }
    var ttftIdle = ttftFor(prompt);
    var ttftLoaded = ttftFor(prompt * bpr);

    // ---- 端到端时延 / 吞吐 ----
    var totalIdle = ttftIdle + output * stepB1;
    var totalLoaded = ttftLoaded + output * step;
    var aggTPS = batch / step;
    var rps = aggTPS / output;
    var prefillTPS = tflopsTot / (2 * activeB * 1e9);

    // ---- 功耗 ----
    var tdpSum = totalGPUs * gpu.tdpW;
    var platformW = tdpSum * K.platformPowerRatio;
    var idleW = tdpSum * K.idlePowerRatio + platformW;
    var avgW = tdpSum * K.decodePowerRatio + platformW;
    var peakW = tdpSum + platformW;
    var jPerToken = avgW / aggTPS;

    // ---- 显存结论 ----
    var util = best.mempg / usable;
    var weightsFit = best.wpg + best.wpg * K.activRatio + K.runtimeGB <= usable;
    if (!best.fits) {
      var rec = minCards(m, gpu, p, ctx, false);
      var recPow2 = minCards(m, gpu, p, ctx, true);
      var recInt4 = minCards(m, gpu, 'int4', ctx, true);
      var sug;
      if (weightsFit) {
        sug = '瓶颈是 KV 缓存：减小上下文长度或并发数，或增加卡数(TP/PP 分摊 KV)';
      } else if (rec.n === recPow2.n && rec.tp === recPow2.tp && rec.pp === recPow2.pp) {
        sug = '权重放不下：当前精度需 ≥ ' + recPow2.n + ' 卡（TP' + recPow2.tp + '×PP' + recPow2.pp + '）；INT4(AWQ/GPTQ) 常规配置需 ≥ ' + recInt4.n + ' 卡（TP' + recInt4.tp + '×PP' + recInt4.pp + '）；或换更大显存的 GPU（H200/B200）';
      } else {
        sug = '权重放不下：当前精度理论最小需 ≥ ' + rec.n + ' 卡（TP' + rec.tp + '×PP' + rec.pp + '，非常规）；' +
          '常规配置建议 ≥ ' + recPow2.n + ' 卡（TP' + recPow2.tp + '×PP' + recPow2.pp + '）；' +
          'INT4(AWQ/GPTQ) 常规配置需 ≥ ' + recInt4.n + ' 卡（TP' + recInt4.tp + '×PP' + recInt4.pp + '）；或换更大显存的 GPU（H200/B200）';
      }
      issues.push({ level: 'error', text: '显存不足：单卡需 ' + best.mempg.toFixed(1) + ' GB，可用 ' + usable.toFixed(1) + ' GB（' + gpu.name + ' ×' + totalGPUs + '）', sug: sug });
    } else if (util > 0.9) {
      issues.push({ level: 'warn', text: '显存余量 < 10%（利用率 ' + (util * 100).toFixed(0) + '%），长上下文/并发波动有 OOM 风险', sug: '预留更多 KV 余量：减小上下文、降低并发或增加卡数' });
    }

    var hasError = issues.some(function (i) { return i.level === 'error'; });
    var status = hasError ? 'fail' : (util > 0.9 ? 'tight' : 'ok');

    return {
      status: status,
      issues: issues,
      strategy: { tp: tp, pp: pp, dp: dp, bpr: bpr, crossNode: best.crossNode, totalGPUs: totalGPUs, commGBs: commGBs, intraGBs: intraGBs, interGBs: interGBs, perReplicaGPUs: tp * pp },
      mem: {
        weightGB: weightGB,
        kvPerTokenB: kvPerTokenB,
        kvSeqPerGPU: best.kvseq,
        kvPerGPU: best.kvpg,
        perGPU: { weights: best.wpg, kv: best.kvpg, activ: best.wpg * K.activRatio, runtime: K.runtimeGB, total: best.mempg, usable: usable, util: util },
        kvTotalGB: kvPerTokenB * ctx * batch / GiB,
        kvPerSeqTotalGB: kvPerTokenB * ctx / GiB
      },
      perf: {
        ttftIdleMs: ttftIdle * 1000,
        ttftLoadedMs: ttftLoaded * 1000,
        stepMs: step * 1000,
        stepB1Ms: stepB1 * 1000,
        perReqTPS: 1 / step,
        perReqTPSIdle: 1 / stepB1,
        totalIdleMs: totalIdle * 1000,
        totalLoadedMs: totalLoaded * 1000,
        aggTPS: aggTPS,
        rps: rps,
        prefillTPS: prefillTPS,
        memTimeMs: memTime * 1000,
        computeTimeMs: computeTime * 1000,
        commMs: commMs,
        activeB: activeB
      },
      power: {
        totalGPUs: totalGPUs, tdpW: gpu.tdpW,
        idleW: idleW, avgW: avgW, peakW: peakW, peakPueW: peakW * K.pue,
        jPerToken: jPerToken, tokensPerKWh: 3.6e6 / jPerToken
      },
      knobs: K
    };
  }

  export { estimate, kvPerTokenBytes, minCards, tflopsFor };
  export const knobs = K;
  export const prec = PREC;
