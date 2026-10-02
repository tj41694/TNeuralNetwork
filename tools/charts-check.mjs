// tools/charts-check.mjs —— 用假 canvas 把 2D 图表代码跑一遍真实数据。
//
// 目的：浏览器里最难查的就是几何数学出错后"画出一片空白"或"坐标变 NaN"。
// 这里用真实 run 的 scalars 数据驱动 TimeSeriesChart，并断言所有绘制坐标都是有限数。
//
// 用法: node tools/charts-check.mjs <origin> <runName>

import {
  JsonlSource,
} from '../web/protocol.js';
import { TimeSeriesChart } from '../web/charts.js';

// charts.js 是浏览器代码，只用到 window.devicePixelRatio 这一处宿主环境，
// 这里补一个最小替身，就能在 Node 下真实驱动渲染逻辑。
globalThis.window = { devicePixelRatio: 1 };

const [originArg, runArg] = process.argv.slice(2);
if (!originArg || !runArg) {
  console.error('用法: node tools/charts-check.mjs <origin> <runName>');
  process.exit(2);
}
const origin = originArg.replace(/\/$/, '');

let passed = 0;
let failed = 0;
function check(name, ok, detail = '') {
  if (ok) {
    passed += 1;
    console.log(`  PASS  ${name}${detail ? '  (' + detail + ')' : ''}`);
  } else {
    failed += 1;
    console.log(`  FAIL  ${name}${detail ? '  (' + detail + ')' : ''}`);
  }
}

// 一个足够用的假 2D context：记录所有几何调用，任何非有限数都记成问题
function makeFake() {
  const bad = [];
  const calls = { fillText: 0, stroke: 0, moveTo: 0, lineTo: 0 };
  const nums = (name, args) => {
    calls[name] = (calls[name] || 0) + 1;
    for (const a of args) {
      if (typeof a === 'number' && !Number.isFinite(a)) bad.push(`${name}(${args.join(',')})`);
    }
  };
  const ctx = {
    canvas: null,
    set fillStyle(v) { if (typeof v === 'string' && v.includes('NaN')) bad.push(`fillStyle=${v}`); },
    get fillStyle() { return '#000'; },
    set strokeStyle(v) { if (typeof v === 'string' && v.includes('NaN')) bad.push(`strokeStyle=${v}`); },
    get strokeStyle() { return '#000'; },
    lineWidth: 1,
    globalAlpha: 1,
    font: '',
    textAlign: 'left',
    textBaseline: 'top',
    clearRect: (...a) => nums('clearRect', a),
    fillRect: (...a) => nums('fillRect', a),
    beginPath: () => {},
    moveTo: (...a) => nums('moveTo', a),
    lineTo: (...a) => nums('lineTo', a),
    stroke: () => { calls.stroke += 1; },
    fill: () => {},
    setLineDash: () => {},
    fillText: (t, ...a) => { calls.fillText += 1; nums('fillText', a); },
    measureText: (t) => ({ width: String(t).length * 6 }),
  };
  const canvas = {
    width: 0,
    height: 0,
    getContext: () => ctx,
    getBoundingClientRect: () => ({ width: 900, height: 180 }),
  };
  return { canvas, ctx, bad, calls };
}

async function main() {
  const src = new JsonlSource(`${origin}/runs/${encodeURIComponent(runArg)}/scalars.jsonl`);
  await src.poll();
  const rows = src.items;
  check('拿到真实数据', rows.length > 100, `rows=${rows.length}`);

  const charts = [
    {
      name: 'loss(对数)',
      canvas: makeFake(),
      logY: true,
      series: [{ key: 'loss', label: 'loss', color: '#5ac8fa', valueOf: (r) => r.loss, ema: true, emaColor: '#ffd166' }],
    },
    {
      name: 'lr(线性)',
      canvas: makeFake(),
      logY: false,
      series: [{ key: 'lr', label: 'lr', color: '#a78bfa', valueOf: (r) => r.lr }],
    },
    {
      name: '范数(对数,双系列)',
      canvas: makeFake(),
      logY: true,
      series: [
        { key: 'gradNorm', label: 'grad', color: '#f87171', valueOf: (r) => r.gradNorm },
        { key: 'weightNorm', label: 'weight', color: '#34d399', valueOf: (r) => r.weightNorm },
      ],
    },
    {
      name: '更新比(对数)',
      canvas: makeFake(),
      logY: true,
      series: [{ key: 'updateRatio', label: 'ratio', color: '#fbbf24', valueOf: (r) => r.updateRatio }],
    },
  ];

  for (const c of charts) {
    const chart = new TimeSeriesChart(c.canvas.canvas, {
      series: c.series,
      logY: c.logY,
      totalSteps: rows.length ? rows[rows.length - 1].step : 0,
    });
    chart.setRows(rows);
    chart.render();
    check(`${c.name}: 坐标全部有限`, c.canvas.bad.length === 0,
      c.canvas.bad.length ? c.canvas.bad.slice(0, 3).join(' | ') : `strokes=${c.canvas.calls.stroke}`);
    check(`${c.name}: 确实画了内容`, c.canvas.calls.moveTo > 0 && c.canvas.calls.fillText > 0,
      `moveTo=${c.canvas.calls.moveTo} fillText=${c.canvas.calls.fillText}`);
  }

  // 空数据、单点、全常量、含 0/负值（对数轴）这些边界不能把图表搞崩
  const edgeCases = [
    { name: '空数据', rows: [] },
    { name: '单点', rows: [{ step: 1, loss: 1.5, lr: 0.1, gradNorm: 1, weightNorm: 1, updateRatio: 0.1 }] },
    {
      name: '全常量',
      rows: new Array(50).fill(0).map((_, i) => ({ step: i + 1, loss: 2, lr: 0.1, gradNorm: 0, weightNorm: 0, updateRatio: 0 })),
    },
    {
      name: '含 0 与负值',
      rows: [
        { step: 1, loss: 0, lr: 0, gradNorm: -1, weightNorm: 0, updateRatio: -5 },
        { step: 2, loss: -3, lr: 0.1, gradNorm: 2, weightNorm: 1, updateRatio: 0 },
      ],
    },
  ];
  for (const ec of edgeCases) {
    const fake = makeFake();
    const chart = new TimeSeriesChart(fake.canvas, {
      series: [
        { key: 'loss', label: 'loss', color: '#fff', valueOf: (r) => r.loss, ema: true },
        { key: 'gradNorm', label: 'grad', color: '#fff', valueOf: (r) => r.gradNorm },
      ],
      logY: true,
      totalSteps: 0,
    });
    chart.setRows(ec.rows);
    chart.render();
    check(`边界「${ec.name}」不产生 NaN 坐标`, fake.bad.length === 0,
      fake.bad.slice(0, 2).join(' | '));
  }

  console.log(`\n结果: ${passed} 通过, ${failed} 失败`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('脚本异常:', err);
  process.exit(1);
});
