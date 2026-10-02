// web/app.js —— 把增量拉取、2D canvas 折线、WebGL2 热力图接起来。
// 只做展示，不做任何持久化。
import {
  JsonlSource,
  RecordSource,
  fetchJson,
  parseHistogramHeader,
  decodeHistogramRecord,
  parseActivationHeader,
  decodeActivationRecord,
  parseProbeInputs,
} from './protocol.js';
import { TimeSeriesChart, drawActivations } from './charts.js';
import { HistogramHeatmap } from './gl.js';

const el = (id) => document.getElementById(id);

const state = {
  origin: location.origin,
  run: null,
  meta: null,
  status: null,
  scalars: null,
  hist: null,
  act: null,
  probeInputs: null,
  probeIndex: 0,
  histPeak: [],
  histEdge: [],
  charts: [],
  heatmap: null,
  timer: null,
  interval: 300,
  failures: 0,
  lastStateFetch: 0,
  lastDraw: 0,
};

function setPill(text, cls) {
  const p = el('status');
  p.textContent = text;
  p.className = `pill ${cls || ''}`;
}

function schedule(delay) {
  if (state.timer) clearTimeout(state.timer);
  const hidden = document.hidden;
  const wait = delay !== undefined ? delay : state.interval;
  state.timer = setTimeout(pollAll, hidden ? Math.max(wait, 4000) : wait);
}

function buildCharts(meta) {
  const fields = new Map((meta.scalarFields || []).map((f) => [f.key, f]));
  const label = (k, d) => (fields.get(k) ? fields.get(k).label : d);
  const isLog = (k, d) => (fields.has(k) ? !!fields.get(k).logScale : d);

  const defs = [
    {
      canvas: el('loss'),
      log: isLog('loss', true),
      series: [
        {
          key: 'loss',
          label: label('loss', 'loss'),
          color: '#5ac8fa',
          valueOf: (r) => r.loss,
          ema: true,
          emaAlpha: 0.03,
          emaColor: '#ffd166',
        },
      ],
    },
    {
      canvas: el('lr'),
      log: isLog('lr', false),
      series: [{ key: 'lr', label: label('lr', 'lr'), color: '#a78bfa', valueOf: (r) => r.lr }],
    },
    {
      canvas: el('norm'),
      log: isLog('gradNorm', true),
      series: [
        { key: 'gradNorm', label: label('gradNorm', 'grad'), color: '#f87171', valueOf: (r) => r.gradNorm },
        { key: 'weightNorm', label: label('weightNorm', 'weight'), color: '#34d399', valueOf: (r) => r.weightNorm },
      ],
    },
    {
      canvas: el('ratio'),
      log: isLog('updateRatio', true),
      series: [
        { key: 'updateRatio', label: label('updateRatio', 'update ratio'), color: '#fbbf24', valueOf: (r) => r.updateRatio },
      ],
    },
  ];

  state.charts = defs.map(
    (d) =>
      new TimeSeriesChart(d.canvas, {
        series: d.series,
        logY: d.log,
        totalSteps: (meta.hyperparams && meta.hyperparams.totalSteps) || 0,
      }),
  );
}

function setupHeatmap(meta) {
  const logging = meta.logging || {};
  const totalSteps = (meta.hyperparams && meta.hyperparams.totalSteps) || 0;
  const histEvery = logging.histEvery || 0;
  if (!histEvery) {
    el('histHint').textContent = '本 run 未开启直方图采样';
    return;
  }
  const totalRecords = Math.floor(totalSteps / histEvery);
  const layerCount = ((meta.network && meta.network.layers) || []).length;
  const layers = (meta.network && meta.network.layers) || [];
  const sel = el('histLayer');
  sel.innerHTML = '';
  layers.forEach((l, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `L${i}  ${l.in}→${l.out}`;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    if (state.heatmap) state.heatmap.setLayer(Number(sel.value));
  };
  state.heatmap = new HistogramHeatmap(el('hist'), {
    bins: logging.histBins || 64,
    layerCount,
    totalRecords: Math.max(1, totalRecords),
  });
  state.histPeak = new Array(layerCount).fill(0);
  state.histEdge = new Array(layerCount).fill(0);
}

// 分箱范围是每层固定死在文件头里的；超出范围的值会被并进边缘 bin。
// 边缘 bin 里同时装着"合法尾巴"和"被截断的极值"，所以不能只看"有没有权重越界"——
// 这里同时跟踪每层 |w| 峰值和边缘 bin 的质量占比，只有后者明显时才当成真问题。
const EDGE_MATERIAL = 0.05;

function updateHistHint() {
  const header = state.hist && state.hist.header;
  if (!header || !state.hist.records.length) return;
  const peaks = state.histPeak || [];
  const edges = state.histEdge || [];
  const node = el('histHint');
  const base = `${state.hist.records.length} 条记录`;

  const worst = edges.reduce((a, e, i) => (e > a.e ? { e, i } : a), { e: 0, i: 0 });
  const over = header.layers
    .map((l, i) => ({ i, p: peaks[i] || 0, range: l.max }))
    .filter((x) => x.p > x.range + 1e-6);

  if (worst.e > EDGE_MATERIAL) {
    node.textContent = `${base} · ⚠ 截断明显：L${worst.i} 边缘 bin 占 ${(worst.e * 100).toFixed(1)}%，建议加大 --hist-range`;
    node.className = 'hint warn';
  } else if (over.length) {
    const desc = over.map((x) => `L${x.i} 峰值 ${x.p.toFixed(2)}>±${x.range}`).join('，');
    node.textContent = `${base} · 轻微截断（${desc}；边缘 bin 占 ${(worst.e * 100).toFixed(1)}%，可忽略）`;
    node.className = 'hint';
  } else {
    const top = peaks.reduce((a, p, i) => (p > a.p ? { p, i } : a), { p: 0, i: 0 });
    node.textContent = `${base} · 未截断（最大 |w| = L${top.i} ${top.p.toFixed(2)}）`;
    node.className = 'hint';
  }
}

// 把新记录推给热力图，同时更新每层的 |w| 峰值与边缘 bin 堆积率
function pushHistRecords(from) {
  if (!state.heatmap) return;
  for (let i = from; i < state.hist.records.length; i += 1) {
    const rec = state.hist.records[i];
    state.heatmap.pushRecord(rec);
    rec.layers.forEach((l, li) => {
      const p = Math.max(Math.abs(l.min), Math.abs(l.max));
      if (!(state.histPeak[li] >= p)) state.histPeak[li] = p;
      let tot = 0;
      for (let k = 0; k < l.counts.length; k += 1) tot += l.counts[k];
      const edge = tot > 0 ? (l.counts[0] + l.counts[l.counts.length - 1]) / tot : 0;
      if (!(state.histEdge[li] >= edge)) state.histEdge[li] = edge;
    });
  }
  updateHistHint();
}

function setupProbeSelector(meta) {
  const sel = el('probeSel');
  const probes = meta.probes || [];
  sel.innerHTML = '';
  probes.forEach((p, i) => {
    const o = document.createElement('option');
    o.value = String(i);
    o.textContent = `probe ${i} · label ${p.label}`;
    sel.appendChild(o);
  });
  sel.onchange = () => {
    state.probeIndex = Number(sel.value);
    drawActivationsPanel();
  };
  state.probeIndex = 0;
}

function onReset() {
  for (const ch of state.charts) {
    ch.yMin = null;
    ch.yMax = null;
  }
  if (state.heatmap) state.heatmap.reset();
  state.histPeak = [];
  state.histEdge = [];
  el('warn').textContent = '检测到文件被截断或换 run，已重置';
}

function updateHeader() {
  const st = state.status || {};
  const meta = state.meta || {};
  const total = (meta.hyperparams && meta.hyperparams.totalSteps) || st.totalSteps || 0;
  const rows = (state.scalars && state.scalars.items) || [];
  const last = rows.length ? rows[rows.length - 1] : null;

  el('progress').textContent = last
    ? `step ${last.step} / ${total} · ${((100 * last.step) / (total || 1)).toFixed(1)}%`
    : '等待数据…';

  const nameMap = {
    running: ['训练中', 'ok'],
    finished: ['已完成', 'done'],
    crashed: ['已崩溃', 'bad'],
    interrupted: ['已中断', 'warn'],
  };
  const entry = nameMap[st.state] || [st.state || '未知', ''];
  setPill(entry[0], entry[1]);

  const stale = st.heartbeatMs ? Date.now() - st.heartbeatMs : null;
  if (st.state === 'running' && stale !== null && stale > 15000) {
    setPill('疑似停滞', 'warn');
  }

  const bad = (state.scalars && state.scalars.badLines) || 0;
  el('warn').textContent = bad > 0 ? `已跳过 ${bad} 行无法解析的数据` : '';
}

function drawActivationsPanel() {
  const recs = (state.act && state.act.records) || [];
  if (!recs.length) {
    drawActivations(el('act'), {});
    return;
  }
  const meta = state.meta || {};
  const probes = meta.probes || [];
  const lastStep = recs[recs.length - 1].step;
  const group = recs.filter((r) => r.step === lastStep);
  const idx = Math.min(state.probeIndex, Math.max(0, group.length - 1));
  const rec = group[idx] || recs[recs.length - 1];
  const info = probes[idx] || {};
  drawActivations(el('act'), {
    input: state.probeInputs ? state.probeInputs.pixels[idx] : null,
    width: state.probeInputs ? state.probeInputs.width : 0,
    layers: rec.layers,
    sizes: state.act.header ? state.act.header.sizes : null,
    label: info.label,
    step: rec.step,
  });
}

function drawAll(force) {
  const now = performance.now();
  if (!force && now - state.lastDraw < 100) return; // 合并同一帧内的多次触发
  state.lastDraw = now;
  for (const ch of state.charts) ch.render();
  if (state.heatmap) state.heatmap.render();
  drawActivationsPanel();
}

async function refreshState() {
  if (!state.run) return;
  const now = performance.now();
  if (now - state.lastStateFetch < 2000) return;
  state.lastStateFetch = now;
  try {
    const st = await fetchJson(`${state.origin}/api/state?run=${encodeURIComponent(state.run)}`);
    if (st.status) state.status = st.status;
  } catch {
    // 状态拿不到不影响曲线，忽略
  }
}

async function pollAll() {
  if (!state.scalars) return;
  try {
    const [a, b, c] = await Promise.all([state.scalars.poll(), state.hist.poll(), state.act.poll()]);
    state.failures = 0;

    if (a.reset || b.reset || c.reset) onReset();

    if (b.added > 0 && state.heatmap) {
      pushHistRecords(state.hist.records.length - b.added);
    }

    if (a.added > 0) {
      for (const ch of state.charts) ch.setRows(state.scalars.items);
    }

    await refreshState();
    updateHeader();
    drawAll();

    const gotNew = a.added + b.added + c.added > 0;
    const running = (state.status && state.status.state) === 'running';
    state.interval = gotNew ? 300 : running ? 800 : 3000;
  } catch (err) {
    state.failures += 1;
    state.interval = Math.min(5000, 800 * state.failures);
    setPill('读取失败', 'bad');
    el('warn').textContent = String((err && err.message) || err);
  }
  schedule();
}

async function loadRun(name) {
  const st = await fetchJson(`${state.origin}/api/state?run=${encodeURIComponent(name)}`);
  if (!st.meta) {
    throw new Error('meta.json 还没生成，稍后重试');
  }
  state.run = name;
  state.meta = st.meta;
  state.status = st.status;
  state.lastStateFetch = performance.now();

  const base = `${state.origin}/runs/${encodeURIComponent(name)}`;
  state.scalars = new JsonlSource(`${base}/scalars.jsonl`);
  state.hist = new RecordSource(`${base}/histograms.bin`, {
    parseHeader: parseHistogramHeader,
    decodeRecord: decodeHistogramRecord,
  });
  state.act = new RecordSource(`${base}/activations.bin`, {
    parseHeader: parseActivationHeader,
    decodeRecord: decodeActivationRecord,
  });

  buildCharts(state.meta);
  setupHeatmap(state.meta);
  setupProbeSelector(state.meta);
  el('origin').textContent = state.origin;

  await Promise.all([state.scalars.poll(), state.hist.open(), state.act.open()]);
  if (state.hist.records.length && state.heatmap) {
    pushHistRecords(0);
  }

  // probe_inputs.bin 是一次性写入的，取一次即可
  try {
    const res = await fetch(`${base}/probe_inputs.bin`, { cache: 'no-store' });
    if (res.ok) state.probeInputs = parseProbeInputs(new Uint8Array(await res.arrayBuffer()));
  } catch {
    state.probeInputs = null;
  }

  for (const ch of state.charts) ch.setRows(state.scalars.items);
  updateHeader();
  drawAll(true);
}

async function boot() {
  const sel = el('run');
  try {
    const list = await fetchJson(`${state.origin}/api/runs`);
    sel.innerHTML = '';
    for (const r of list.runs) {
      const o = document.createElement('option');
      o.value = r.name;
      const st = r.status ? ` · ${r.status.state} · step ${r.status.lastStep}` : '';
      o.textContent = r.name + st;
      sel.appendChild(o);
    }
    if (!list.runs.length) {
      el('warn').textContent = 'runs/ 下还没有任何 run';
      return;
    }
    const wanted = new URLSearchParams(location.search).get('run');
    const name = wanted && list.runs.some((r) => r.name === wanted) ? wanted : list.runs[0].name;
    sel.value = name;
    sel.onchange = async () => {
      location.search = `?run=${encodeURIComponent(sel.value)}`;
    };
    await loadRun(name);
    schedule(0);
  } catch (err) {
    setPill('无法连接', 'bad');
    el('warn').textContent = String((err && err.message) || err);
    schedule(2000);
  }
}

el('fullAxis').addEventListener('change', (e) => {
  for (const ch of state.charts) ch.fullAxis = e.target.checked;
  drawAll(true);
});

document.addEventListener('visibilitychange', () => {
  if (!document.hidden) {
    state.lastStateFetch = 0;
    schedule(0);
  }
});

if (typeof ResizeObserver !== 'undefined') {
  const ro = new ResizeObserver(() => drawAll(true));
  for (const c of document.querySelectorAll('canvas')) ro.observe(c);
}

boot();
