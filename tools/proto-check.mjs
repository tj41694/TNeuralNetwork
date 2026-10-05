// tools/proto-check.mjs —— 用真实 HTTP 服务与真实数据文件校验增量拉取协议。
//
// 覆盖 docs/telemetry-plan.md §9 里最容易出错的几条路径：
//   206 正常追加、416 无新数据、416 掩盖的文件截断、尾部半行、服务器忽略 Range 的 200 降级。
//
// 用法:
//   node tools/proto-check.mjs <origin> <runName> <runsDir> [staticOrigin]
//     <origin>       C++ 服务地址，如 http://127.0.0.1:5000
//     <runName>      要校验的 run 目录名
//     <runsDir>      本地 runs 根目录（用于造一个临时 run 做破坏性测试，不动真数据）
//     [staticOrigin] 一个不支持 Range 的静态服务器地址（如 python -m http.server），
//                    用来验证 200 降级路径
//
// 例:
//   node tools/proto-check.mjs http://127.0.0.1:5000 live_20261002-162744 ../runs

import fs from 'node:fs/promises';
import path from 'node:path';
import {
  JsonlSource,
  RecordSource,
  parseHistogramHeader,
  decodeHistogramRecord,
  parseActivationHeader,
  decodeActivationRecord,
} from '../web/protocol.js';

const [originArg, runArg, runsDirArg, staticOriginArg] = process.argv.slice(2);
if (!originArg || !runArg || !runsDirArg) {
  console.error('用法: node tools/proto-check.mjs <origin> <runName> <runsDir> [staticOrigin]');
  process.exit(2);
}
const origin = originArg.replace(/\/$/, '');
const run = runArg;
const runsDir = path.resolve(runsDirArg);
const base = `${origin}/runs/${encodeURIComponent(run)}`;

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

async function fetchBytes(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return new Uint8Array(await res.arrayBuffer());
}

async function fetchText(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.text();
}

async function scalarsChecks(label, srcBase, expectLines) {
  const src = new JsonlSource(`${srcBase}/scalars.jsonl`);
  await src.poll();
  check(`${label}: JSONL 行数与文件一致`, src.items.length === expectLines,
    `client=${src.items.length} file=${expectLines}`);
  check(`${label}: JSONL 无坏行`, src.badLines === 0, `badLines=${src.badLines}`);
  const keys = ['step', 'samplesSeen', 'lr', 'loss', 'gradNorm', 'weightNorm', 'updateRatio'];
  check(`${label}: 字段完整且为有限数`,
    src.items.length > 0 && keys.every((k) => Number.isFinite(src.items[0][k])));
  check(`${label}: step 从 1 起严格递增`,
    src.items.every((r, i) => (i === 0 ? r.step === 1 : r.step === src.items[i - 1].step + 1)));
  const again = await src.poll();
  check(`${label}: 二次轮询无新数据也不重复`, again.added === 0 && src.items.length === expectLines,
    `added=${again.added}`);
  return src;
}

async function main() {
  console.log(`origin = ${origin}\nrun    = ${run}\nrunsDir= ${runsDir}\n`);

  // ---------- 1. 只读校验真实 run ----------
  const text = await fetchText(`${base}/scalars.jsonl`);
  const fileLines = text.split('\n').filter((l) => l.trim() !== '').length;
  await scalarsChecks('真实 run', base, fileLines);

  const histBytes = await fetchBytes(`${base}/histograms.bin`);
  const hist = new RecordSource(`${base}/histograms.bin`, {
    parseHeader: parseHistogramHeader,
    decodeRecord: decodeHistogramRecord,
  });
  const histOpened = await hist.open();
  check('histograms.bin 头部可解析',
    histOpened && hist.header.magic === 'TNH1' && hist.recordBytes > 0,
    `bins=${hist.header && hist.header.bins} recordBytes=${hist.recordBytes} headerBytes=${hist.headerBytes}`);
  const expectHist = Math.floor((histBytes.length - hist.headerBytes) / hist.recordBytes);
  check('histograms.bin 记录数一致', hist.records.length === expectHist,
    `client=${hist.records.length} file=${expectHist}`);
  const histEvery = hist.records.length > 1 ? hist.records[1].step - hist.records[0].step : 0;
  check('直方图 step 间隔恒定', hist.records.every((r, i) =>
    i === 0 || r.step === hist.records[i - 1].step + histEvery), `every=${histEvery}`);
  check('直方图 counts 总和为正', hist.records.length > 0 && hist.records[0].layers.every((l) => {
    let sum = 0;
    for (const v of l.counts) sum += v;
    return sum > 0;
  }));
  // counts 之和必须等于该层的参数个数（权重 in×out + 偏置 out）。
  // 这是直方图最硬的正确性证据：既证明分箱没丢值，也证明没重复计数。
  const paramCountOk = (rec) => rec.layers.every((l, i) => {
    const meta = hist.header.layers[i];
    let sum = 0;
    for (const v of l.counts) sum += v;
    return sum === meta.in * meta.out + meta.out;
  });
  check('首条直方图 counts 之和 = 该层参数个数', hist.records.length > 0 && paramCountOk(hist.records[0]));
  check('末条直方图 counts 之和 = 该层参数个数', hist.records.length > 0 && paramCountOk(hist.records[hist.records.length - 1]));

  // 分箱范围是**每层各自固定**的（写在该层文件头的 min/max 里）。边缘 bin 里同时装着
  // "合法尾巴"和"被截断的极值"，所以按**堆积质量占比**判断严重程度，而不是"有没有任何一个
  // 权重越界"—— 输出层那几个离群权重不值得为它牺牲全 run 的分辨率。
  if (hist.records.length > 0) {
    const L = hist.header.layers.length;
    const peaks = new Array(L).fill(0);
    const edgeMax = new Array(L).fill(0);
    for (const rec of hist.records) {
      rec.layers.forEach((l, i) => {
        const p = Math.max(Math.abs(l.min), Math.abs(l.max));
        if (p > peaks[i]) peaks[i] = p;
        let tot = 0;
        let edge = 0;
        for (let k = 0; k < l.counts.length; k += 1) {
          tot += l.counts[k];
          if (k === 0 || k === l.counts.length - 1) edge += l.counts[k];
        }
        const share = tot > 0 ? edge / tot : 0;
        if (share > edgeMax[i]) edgeMax[i] = share;
      });
    }
    const ranges = hist.header.layers.map((l) => l.max);
    const allSame = ranges.every((r) => r === ranges[0]);
    const rangeDesc = allSame ? `±${ranges[0]}` : `[${ranges.map((r) => `±${r}`).join(', ')}]`;
    const worst = edgeMax.reduce((a, e, i) => (e > a.e ? { e, i } : a), { e: 0, i: 0 });
    const summary = `各层峰值 ${peaks.map((p) => p.toFixed(2)).join('/')}`
      + `，最大边缘堆积 ${(worst.e * 100).toFixed(2)}%（L${worst.i}）`;
    if (worst.e > 0.05) {
      console.log(`  WARN  分箱范围 ${rangeDesc} 截断明显：${summary}；`
        + '换更大的 --hist-range（可按层给，如 1,1.5,1.5,2）可以改善');
    } else {
      console.log(`  INFO  分箱范围 ${rangeDesc} 无实质截断：${summary}`);
    }
  }

  const actBytes = await fetchBytes(`${base}/activations.bin`);
  const act = new RecordSource(`${base}/activations.bin`, {
    parseHeader: parseActivationHeader,
    decodeRecord: decodeActivationRecord,
  });
  await act.open();
  const expectAct = Math.floor((actBytes.length - act.headerBytes) / act.recordBytes);
  check('activations.bin 记录数一致', act.records.length === expectAct,
    `client=${act.records.length} file=${expectAct}`);
  const uniqueSteps = new Set(act.records.map((r) => r.step)).size;
  check('激活记录按 step 分组', act.records.every((r, i) =>
    i === 0 || r.step >= act.records[i - 1].step), `steps=${uniqueSteps}`);
  check('激活值均为有限数', act.records.length > 0 &&
    act.records[0].layers.every((vals) => vals.every((v) => Number.isFinite(v))));

  // ---------- 2. 破坏性测试：临时 run，不动真实数据 ----------
  const tmpName = `_proto_tmp_${process.pid}`;
  const tmpDir = path.join(runsDir, tmpName);
  await fs.mkdir(tmpDir, { recursive: true });
  const scalarsPath = path.join(tmpDir, 'scalars.jsonl');
  const sample = text.split('\n').filter((l) => l.trim() !== '').slice(0, 40).join('\n') + '\n';
  await fs.writeFile(scalarsPath, sample);
  await fs.copyFile(path.join(runsDir, run, 'status.json'), path.join(tmpDir, 'status.json'));

  const tmpBase = `${origin}/runs/${encodeURIComponent(tmpName)}`;
  const tmpSrc = new JsonlSource(`${tmpBase}/scalars.jsonl`);
  await tmpSrc.poll();
  check('临时 run: 初始行数', tmpSrc.items.length === 40, `items=${tmpSrc.items.length}`);

  // 2a. 追加半行 -> 不能被解析，也不能丢
  const rows = new Array(3).fill(0).map((_, i) => JSON.stringify({
    step: 1000 + i, samplesSeen: i, tMs: i, wallMs: i,
    lr: 0.1, loss: i + 1, gradNorm: 1, weightNorm: 1, updateRatio: 0.5,
  }));
  await fs.appendFile(scalarsPath, rows[0].slice(0, 25));
  await tmpSrc.poll();
  check('半行不解析（保留在尾部）', tmpSrc.items.length === 40 && tmpSrc.badLines === 0,
    `items=${tmpSrc.items.length} bad=${tmpSrc.badLines}`);

  // 2b. 补齐半行 -> 只新增这一条
  await fs.appendFile(scalarsPath, rows[0].slice(25) + '\n');
  const inc = await tmpSrc.poll();
  check('补齐后只追加一条', tmpSrc.items.length === 41 && inc.added === 1, `added=${inc.added}`);

  // 2c. 再追加两条 -> 一次拿到两条
  await fs.appendFile(scalarsPath, rows[1] + '\n' + rows[2] + '\n');
  const inc2 = await tmpSrc.poll();
  check('批量追加一次拿到两条', tmpSrc.items.length === 43 && inc2.added === 2, `added=${inc2.added}`);

  // 2d. 截断 -> 必须检测到并整体重置（服务端此时返回 416，因为起点 >= 新长度）
  const truncated = sample.split('\n').slice(0, 10).join('\n') + '\n';
  await fs.writeFile(scalarsPath, truncated);
  const resetRes = await tmpSrc.poll();
  check('截断被检测到并重置', resetRes.reset === true && tmpSrc.items.length === 10,
    `reset=${resetRes.reset} items=${tmpSrc.items.length}`);
  const after = await tmpSrc.poll();
  check('重置后继续增量正常', after.added === 0 && tmpSrc.items.length === 10, `added=${after.added}`);

  // 2e. 追加新内容 -> 重置后的 offset 正确
  await fs.appendFile(scalarsPath, rows[0] + '\n');
  const after2 = await tmpSrc.poll();
  check('重置后追加仍能同步', tmpSrc.items.length === 11 && after2.added === 1, `added=${after2.added}`);

  await fs.rm(tmpDir, { recursive: true, force: true });

  // ---------- 3. 200 降级：服务器忽略 Range ----------
  if (staticOriginArg) {
    const sOrigin = staticOriginArg.replace(/\/$/, '');
    const s = new JsonlSource(`${sOrigin}/${run}/scalars.jsonl`);
    await s.poll();
    check('200 降级: 首次全量', s.items.length === fileLines, `items=${s.items.length}`);
    const p2 = await s.poll();
    check('200 降级: 二次不重复', s.items.length === fileLines && p2.added === 0,
      `items=${s.items.length} added=${p2.added}`);
    const p3 = await s.poll();
    check('200 降级: 三次仍稳定', s.items.length === fileLines && p3.added === 0);
  } else {
    console.log('  SKIP  200 降级（未提供静态服务器地址）');
  }

  console.log(`\n结果: ${passed} 通过, ${failed} 失败`);
  process.exit(failed === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error('脚本异常:', err);
  process.exit(1);
});
