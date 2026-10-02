// web/charts.js —— 2D canvas 图表。文字、坐标轴、折线都在这里，
// 简单面板不需要 WebGL；WebGL2 只用在直方图热力图那种大数据量贴图场景（见 gl.js）。

export function setupCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return { ctx: canvas.getContext('2d'), w, h, dpr };
}

const FLOOR = 1e-12;

function fmt(v, digits = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

// 线性轴的"好看"刻度
function linearTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * mag;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) ticks.push(t);
  return ticks;
}

// 对数轴的十进制刻度
function logTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.floor(lo); e <= Math.ceil(hi); e += 1) {
    if (e >= lo && e <= hi) ticks.push(e);
  }
  return ticks;
}

export class TimeSeriesChart {
  constructor(canvas, opts) {
    this.canvas = canvas;
    this.series = opts.series;
    this.logY = !!opts.logY;
    this.totalSteps = opts.totalSteps || 0;
    this.fullAxis = true;
    this.rows = [];
    this.ema = this.series.map(() => []);
    this.yMin = null;
    this.yMax = null;
    this.locked = false;
  }

  setTotalSteps(total) {
    this.totalSteps = total;
  }

  // 每次都传完整数组：25000 行的重算在本地是毫秒级，换来的是实现简单且不会漂移
  setRows(rows) {
    this.rows = rows;
    const n = rows.length;
    for (let s = 0; s < this.series.length; s += 1) {
      const def = this.series[s];
      if (!def.ema) {
        this.ema[s] = [];
        continue;
      }
      const alpha = def.emaAlpha || 0.05;
      const out = new Array(n);
      let acc = null;
      for (let i = 0; i < n; i += 1) {
        const v = def.valueOf(rows[i]);
        if (Number.isFinite(v)) acc = acc === null ? v : acc + alpha * (v - acc);
        out[i] = acc;
      }
      this.ema[s] = out;
    }
  }

  value(s, i) {
    const def = this.series[s];
    const v = def.valueOf(this.rows[i]);
    if (!Number.isFinite(v)) return null;
    if (this.logY) return v > FLOOR ? Math.log10(v) : null;
    return v;
  }

  emaValue(s, i) {
    const v = this.ema[s] ? this.ema[s][i] : null;
    if (v === null || v === undefined || !Number.isFinite(v)) return null;
    if (this.logY) return v > FLOOR ? Math.log10(v) : null;
    return v;
  }

  render() {
    const { ctx, w, h, dpr } = setupCanvas(this.canvas);
    ctx.clearRect(0, 0, w, h);

    const padL = 56 * dpr;
    const padR = 12 * dpr;
    const padT = 10 * dpr;
    const padB = 22 * dpr;
    const plotW = w - padL - padR;
    const plotH = h - padT - padB;
    if (plotW <= 4 || plotH <= 4) return;

    const rows = this.rows;
    const lastStep = rows.length ? rows[rows.length - 1].step : 0;
    const x1 = Math.max(this.fullAxis ? this.totalSteps : 0, lastStep, 1);

    let dmin = Infinity;
    let dmax = -Infinity;
    for (let s = 0; s < this.series.length; s += 1) {
      for (let i = 0; i < rows.length; i += 1) {
        const v = this.value(s, i);
        if (v === null) continue;
        if (v < dmin) dmin = v;
        if (v > dmax) dmax = v;
        const e = this.emaValue(s, i);
        if (e !== null) {
          if (e < dmin) dmin = e;
          if (e > dmax) dmax = e;
        }
      }
    }
    if (!Number.isFinite(dmin) || !Number.isFinite(dmax)) {
      dmin = 0;
      dmax = 1;
    }
    if (dmax - dmin < 1e-9) {
      dmin -= 0.5;
      dmax += 0.5;
    }

    // Y 轴自适应带迟滞：只在明显超出时扩展，收缩要等数据范围明显变小，
    // 否则每帧 autoscale 会让曲线持续抖动。
    if (this.locked && this.yMin !== null) {
      // 保持锁定范围
    } else if (this.yMin === null) {
      const pad = (dmax - dmin) * 0.06;
      this.yMin = dmin - pad;
      this.yMax = dmax + pad;
    } else {
      const pad = (dmax - dmin) * 0.06;
      if (dmin - pad < this.yMin || dmax + pad > this.yMax) {
        this.yMin = Math.min(this.yMin, dmin - pad);
        this.yMax = Math.max(this.yMax, dmax + pad);
      } else if (dmax - dmin < (this.yMax - this.yMin) * 0.4) {
        this.yMin = dmin - pad;
        this.yMax = dmax + pad;
      }
    }
    const y0 = this.yMin;
    const y1 = this.yMax;

    const sx = (step) => padL + (step / x1) * plotW;
    const sy = (v) => padT + plotH - ((v - y0) / (y1 - y0)) * plotH;

    // 背景与网格
    ctx.fillStyle = '#12161d';
    ctx.fillRect(padL, padT, plotW, plotH);
    ctx.strokeStyle = '#252c38';
    ctx.lineWidth = 1 * dpr;
    ctx.font = `${10 * dpr}px ui-monospace, Consolas, monospace`;
    ctx.fillStyle = '#7c8899';

    let ticks;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    if (this.logY) {
      ticks = logTicks(y0, y1);
      for (const t of ticks) {
        const y = sy(t);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        const label = Math.abs(t) <= 4 ? Math.pow(10, t).toPrecision(2) : `1e${t}`;
        ctx.fillText(label, padL - 6 * dpr, y);
      }
      if (ticks.length === 0) {
        ctx.fillText(`1e${y0.toFixed(1)}`, padL - 6 * dpr, sy(y0));
        ctx.fillText(`1e${y1.toFixed(1)}`, padL - 6 * dpr, sy(y1));
      }
    } else {
      ticks = linearTicks(y0, y1);
      for (const t of ticks) {
        const y = sy(t);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.fillText(fmt(t, 4), padL - 6 * dpr, y);
      }
    }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xticks = linearTicks(0, x1, 6);
    for (const t of xticks) {
      const x = sx(t);
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
      ctx.fillText(String(Math.round(t)), x, padT + plotH + 5 * dpr);
    }

    // 数据：按像素列取 min/max 抽稀，否则几万个点会挤成一条带状噪声
    const cols = Math.max(1, Math.floor(plotW));
    for (let s = 0; s < this.series.length; s += 1) {
      const def = this.series[s];
      const minArr = new Float64Array(cols).fill(NaN);
      const maxArr = new Float64Array(cols).fill(NaN);
      for (let i = 0; i < rows.length; i += 1) {
        const v = this.value(s, i);
        if (v === null) continue;
        const col = Math.min(cols - 1, Math.max(0, Math.floor((sx(rows[i].step) - padL))));
        if (Number.isNaN(minArr[col]) || v < minArr[col]) minArr[col] = v;
        if (Number.isNaN(maxArr[col]) || v > maxArr[col]) maxArr[col] = v;
      }

      // 原始曲线：点数少时直接连线，点多时按列画 min-max 竖线
      ctx.strokeStyle = def.color;
      ctx.lineWidth = (def.width || 1.4) * dpr;
      ctx.globalAlpha = rows.length > cols * 2 ? 0.55 : 1;
      ctx.beginPath();
      let started = false;
      for (let c = 0; c < cols; c += 1) {
        if (Number.isNaN(minArr[c])) continue;
        const x = padL + c + 0.5;
        if (!started) {
          ctx.moveTo(x, sy(maxArr[c]));
          started = true;
        } else {
          ctx.lineTo(x, sy(maxArr[c]));
        }
        ctx.lineTo(x, sy(minArr[c]));
      }
      ctx.stroke();
      ctx.globalAlpha = 1;

      if (def.ema) {
        ctx.strokeStyle = def.emaColor || def.color;
        ctx.lineWidth = 1.8 * dpr;
        ctx.setLineDash([]);
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < rows.length; i += 1) {
          const v = this.emaValue(s, i);
          if (v === null) continue;
          const x = sx(rows[i].step);
          const y = sy(v);
          if (!pen) {
            ctx.moveTo(x, y);
            pen = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // 图例与最新读数
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    let lx = padL + 8 * dpr;
    const ly = padT + 6 * dpr;
    for (let s = 0; s < this.series.length; s += 1) {
      const def = this.series[s];
      const raw = rows.length ? def.valueOf(rows[rows.length - 1]) : null;
      const emaV = rows.length ? (this.ema[s] ? this.ema[s][rows.length - 1] : null) : null;
      const text = def.ema
        ? `${def.label} ${fmt(raw)} · ema ${fmt(emaV)}`
        : `${def.label} ${fmt(raw)}`;
      ctx.fillStyle = def.color;
      ctx.fillRect(lx, ly + 1 * dpr, 8 * dpr, 8 * dpr);
      ctx.fillStyle = '#c7d0dc';
      ctx.fillText(text, lx + 12 * dpr, ly);
      lx += ctx.measureText(text).width + 26 * dpr;
    }
  }
}

// 激活：输入图像 + 每层一条格子带
export function drawActivations(canvas, { input, width, layers, label, step, sizes }) {
  const { ctx, w, h, dpr } = setupCanvas(canvas);
  ctx.clearRect(0, 0, w, h);
  ctx.font = `${11 * dpr}px ui-monospace, Consolas, monospace`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';

  const rowH = 26 * dpr;
  const labelW = 92 * dpr;
  let y = 10 * dpr;

  if (step === undefined || step === null) {
    ctx.fillStyle = '#7c8899';
    ctx.fillText('等待激活采样…', labelW, y + rowH / 2);
    return;
  }

  ctx.fillStyle = '#7c8899';
  ctx.fillText(`step ${step}`, 8 * dpr, y + rowH / 2);
  ctx.fillText(`label ${label}`, 8 * dpr, y + rowH * 1.6 + 8 * dpr);
  y += rowH * 2.2;

  // 输入图像（28x28 灰度，NEAREST 放大）
  if (input && width) {
    const side = Math.round(Math.sqrt(width));
    const box = 56 * dpr;
    const cell = box / side;
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < width; i += 1) {
      if (input[i] < mn) mn = input[i];
      if (input[i] > mx) mx = input[i];
    }
    const span = mx - mn > 1e-9 ? mx - mn : 1;
    for (let r = 0; r < side; r += 1) {
      for (let c = 0; c < side; c += 1) {
        const v = (input[r * side + c] - mn) / span;
        const g = Math.round(255 * v);
        ctx.fillStyle = `rgb(${g},${g},${g})`;
        ctx.fillRect(labelW + c * cell, y + r * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    ctx.fillStyle = '#7c8899';
    ctx.fillText(`输入 ${side}×${side}`, labelW, y + box + 10 * dpr);
    y += box + 24 * dpr;
  }

  // 每层一条带子，每个格子一个神经元，颜色按层内最大绝对值归一化
  for (let l = 0; l < layers.length; l += 1) {
    const values = layers[l];
    const size = sizes && sizes[l] ? sizes[l] : values.length;
    let mx = 0;
    for (let i = 0; i < size; i += 1) mx = Math.max(mx, Math.abs(values[i] || 0));
    const cellW = Math.max(3 * dpr, Math.min(22 * dpr, (w - labelW - 16 * dpr) / size));
    for (let i = 0; i < size; i += 1) {
      const v = mx > 0 ? Math.abs(values[i] || 0) / mx : 0;
      const r = Math.round(40 + 215 * v);
      const g = Math.round(60 + 120 * (1 - Math.abs(v - 0.5) * 2));
      const b = Math.round(90 + 165 * (1 - v));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(labelW + i * cellW, y, Math.max(1, cellW - dpr), rowH - 4 * dpr);
    }
    ctx.fillStyle = '#c7d0dc';
    ctx.fillText(`L${l} · ${size} 神经元`, 8 * dpr, y + rowH / 2);
    y += rowH;
  }
}
