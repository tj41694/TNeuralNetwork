// web/protocol.js —— 增量拉取协议与二进制格式解析。
// 纯逻辑，不依赖 DOM，可以直接在 Node 下跑测试（见 tools/proto-check.mjs）。
//
// 与服务端的约定（详见 docs/telemetry-plan.md §7）：
//   206 -> 只追加本次字节
//   200 -> 服务器忽略了 Range，body 从 0 开始：若比自身 offset 短，说明文件被截断/换过，
//          必须整体重置；否则按自身 offset 切分后追加
//   416 -> 没有新数据（文件长度没变）
//   404 -> 文件还没创建
//
// append-only 的文件前缀永远稳定，这是 offset 语义成立的前提；
// 唯一破坏它的是截断/重写，所以要用"body 比 offset 短"来检测。

export const NO_STORE = { cache: 'no-store' };

export class HttpError extends Error {
  constructor(status, url) {
    super(`HTTP ${status} for ${url}`);
    this.status = status;
    this.url = url;
  }
}

// 从 "bytes 0-4095/227988" 里取出总长度
export function totalFromContentRange(value) {
  if (!value) return null;
  const m = /\/(\d+)\s*$/.exec(value);
  return m ? Number(m[1]) : null;
}

export async function fetchJson(url, options = {}, fetchImpl = fetch) {
  const res = await fetchImpl(url, { ...NO_STORE, ...options });
  if (!res.ok) throw new HttpError(res.status, url);
  return res.json();
}

class ByteSource {
  constructor(url) {
    this.url = url;
    this.offset = 0;
    this.totalBytes = 0;
    this.missing = false;
    // 416 时为了区分"没有新数据"和"文件被截断"要额外探测一次长度；
    // 用一个时间下限把空闲时的额外请求压到每 2 秒最多一次。
    this.lastProbeMs = 0;
  }

  reset() {
    this.offset = 0;
    this.totalBytes = 0;
  }

  // 发一次 Range 请求，并把响应归一化成"本次真正要追加的字节"。
  async fetchRange(fetchImpl = fetch) {
    const res = await fetchImpl(this.url, {
      ...NO_STORE,
      headers: { Range: `bytes=${this.offset}-` },
    });
    const status = res.status;

    if (status === 416) {
      // 416 只说明"起点 >= 文件当前长度"。可能是没有新数据，也可能是文件被截断
      // 或换成了另一个 run —— 两者必须区分开，否则增量同步会永远卡住。
      // 416 响应不带 Content-Range，所以再发一个 1 字节的 Range 请求问出真实长度。
      const now = Date.now();
      if (now - this.lastProbeMs < 2000) {
        return { body: null, didReset: false };
      }
      this.lastProbeMs = now;
      const probe = await fetchImpl(this.url, {
        ...NO_STORE,
        headers: { Range: 'bytes=0-0' },
      });
      if (probe.status === 206) {
        const total = totalFromContentRange(probe.headers.get('Content-Range'));
        await probe.arrayBuffer().catch(() => {});
        if (total !== null) {
          this.totalBytes = total;
          if (total < this.offset) {
            this.reset();
            return { body: null, didReset: true };
          }
        }
      } else if (probe.status === 404) {
        this.missing = true;
        return { body: null, didReset: false };
      } else {
        await probe.arrayBuffer().catch(() => {});
      }
      this.missing = false;
      return { body: null, didReset: false };
    }
    if (status === 404) {
      this.missing = true;
      return { body: null, didReset: false };
    }
    if (status !== 200 && status !== 206) {
      throw new HttpError(status, this.url);
    }
    this.missing = false;

    const total = totalFromContentRange(res.headers.get('Content-Range'));
    let body = new Uint8Array(await res.arrayBuffer());
    let didReset = false;

    if (status === 200) {
      if (body.length < this.offset) {
        // 文件被截断或换成了另一个 run：全量重置
        this.reset();
        didReset = true;
      } else {
        body = body.subarray(this.offset);
      }
    }

    this.offset += body.length;
    if (total !== null) this.totalBytes = total;
    else this.totalBytes = Math.max(this.totalBytes, this.offset);
    return { body, didReset };
  }
}

// JSONL：按 \n 切分，最后一段不完整的内容留在 tail 里等下一轮拼接。
export class JsonlSource extends ByteSource {
  constructor(url) {
    super(url);
    this.items = [];
    this.badLines = 0;
    this.tail = '';
    this.decoder = new TextDecoder('utf-8');
    this.tailCap = 1 << 20;
  }

  reset() {
    super.reset();
    this.items = [];
    this.tail = '';
    this.decoder = new TextDecoder('utf-8');
  }

  async poll(fetchImpl = fetch) {
    let { body, didReset } = await this.fetchRange(fetchImpl);
    if (didReset) {
      this.resetTailOnly();
      // 重置后立刻重取一次：否则要白等一个轮询周期才恢复
      ({ body } = await this.fetchRange(fetchImpl));
    }
    if (!body || body.length === 0) return { added: 0, reset: didReset };
    const before = this.items.length;
    this.consume(body);
    return { added: this.items.length - before, reset: didReset };
  }

  resetTailOnly() {
    this.items = [];
    this.tail = '';
    this.decoder = new TextDecoder('utf-8');
  }

  consume(bytes) {
    // 流式解码：多字节 UTF-8 字符被切在两次响应之间也不会解错
    this.tail += this.decoder.decode(bytes, { stream: true });
    const parts = this.tail.split('\n');
    this.tail = parts.pop();
    if (this.tail.length > this.tailCap) {
      // 这么长都没有换行，只可能是损坏，丢掉避免内存无上限增长
      this.badLines += 1;
      this.tail = '';
    }
    for (const raw of parts) {
      const line = raw.trim();
      if (line === '') continue;
      try {
        this.items.push(JSON.parse(line));
      } catch {
        // 一行坏数据绝不能让整条流断掉
        this.badLines += 1;
      }
    }
  }
}

// 定长记录文件：不足一条记录的部分留在 pending 里。
export class RecordSource extends ByteSource {
  constructor(url, { parseHeader, decodeRecord }) {
    super(url);
    this.parseHeader = parseHeader;
    this.decodeRecord = decodeRecord;
    this.header = null;
    this.headerBytes = 0;
    this.recordBytes = 0;
    this.records = [];
    this.pending = new Uint8Array(0);
    // 首条记录在整个文件里的序号：tail 加载时热力图要靠它对齐列
    this.firstRecordIndex = 0;
  }

  reset() {
    super.reset();
    this.records = [];
    this.pending = new Uint8Array(0);
    // 注意不能清掉 header：截断恢复要靠重新解析头部（换 run 时格式也可能不同），
    // 而 recordBytes 一旦变成 0 后续解码就会除零。
    this.firstRecordIndex = 0;
  }

  // 打开：先读文件头解析出 recordBytes，再决定从哪开始。
  // tailRecords > 0 时直接跳到末尾只取最后若干条（首屏用）。
  async open({ tailRecords = 0, fetchImpl = fetch } = {}) {
    this.records = [];
    this.pending = new Uint8Array(0);

    const res = await fetchImpl(this.url, {
      ...NO_STORE,
      headers: { Range: 'bytes=0-4095' },
    });
    if (res.status === 404) {
      this.missing = true;
      return false;
    }
    if (res.status !== 200 && res.status !== 206) {
      throw new HttpError(res.status, this.url);
    }
    this.missing = false;

    const total = totalFromContentRange(res.headers.get('Content-Range'));
    const buf = new Uint8Array(await res.arrayBuffer());
    if (buf.length < 16) {
      // 文件刚被创建、头部还没写完，等下一轮
      return false;
    }
    this.header = this.parseHeader(buf);
    this.headerBytes = this.header.headerBytes;
    this.recordBytes = this.header.recordBytes;
    this.totalBytes = total === null ? buf.length : total;

    const dataBytes = Math.max(0, this.totalBytes - this.headerBytes);
    const totalRecords = Math.floor(dataBytes / this.recordBytes);
    const keep = tailRecords > 0 ? Math.min(tailRecords, totalRecords) : totalRecords;
    const startByte = this.headerBytes + (totalRecords - keep) * this.recordBytes;
    this.firstRecordIndex = totalRecords - keep;

    if (startByte <= buf.length) {
      // 尾巴就在这个头部响应里（文件很短），直接消费
      this.pending = buf.subarray(startByte);
      this.offset = buf.length;
    } else {
      // 中间那段（可能很长）故意跳过
      this.pending = new Uint8Array(0);
      this.offset = startByte;
    }
    this.decodePending();

    // 头部那次请求只覆盖前 4KB，剩下的要自己追到 EOF，
    // 否则首屏会平白少一截记录。一次 Range 请求就能把剩余部分全取回。
    for (let guard = 0; guard < 256 && this.offset < this.totalBytes; guard += 1) {
      const step = await this.poll(fetchImpl);
      if (step.reset || step.added === 0) break;
    }
    return true;
  }

  async poll(fetchImpl = fetch) {
    if (!this.header || this.recordBytes <= 0) {
      // 页面可能在文件还没生成时就打开了；首次拿到内容时再解析头部，
      // 否则 recordBytes 为 0 会让按记录切分除零。
      const opened = await this.open({ tailRecords: 0, fetchImpl });
      return { added: this.records.length, reset: false, opened };
    }
    const { body, didReset } = await this.fetchRange(fetchImpl);
    if (didReset) {
      // 文件被截断或换 run：头部可能也变了，整体重新打开
      const opened = await this.open({ tailRecords: 0, fetchImpl });
      return { added: this.records.length, reset: true, opened };
    }
    if (!body || body.length === 0) return { added: 0, reset: false };
    const before = this.records.length;
    const merged = new Uint8Array(this.pending.length + body.length);
    merged.set(this.pending, 0);
    merged.set(body, this.pending.length);
    this.pending = merged;
    this.decodePending();
    return { added: this.records.length - before, reset: didReset };
  }

  decodePending() {
    const usable = Math.floor(this.pending.length / this.recordBytes) * this.recordBytes;
    if (usable === 0) return;
    const chunk = this.pending.subarray(0, usable);
    const dv = new DataView(chunk.buffer, chunk.byteOffset, chunk.byteLength);
    for (let off = 0; off < usable; off += this.recordBytes) {
      this.records.push(this.decodeRecord(dv, off, this.header));
    }
    this.pending = this.pending.subarray(usable);
  }
}

function readMagic(buf) {
  return String.fromCharCode(buf[0], buf[1], buf[2], buf[3]);
}

// histograms.bin 头：24 + 16 * 层数 字节
export function parseHistogramHeader(buf) {
  if (buf.length < 24) throw new Error('histogram header too short');
  const magic = readMagic(buf);
  if (magic !== 'TNH1') throw new Error(`bad histogram magic: ${magic}`);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const header = {
    magic,
    version: dv.getUint32(4, true),
    recordBytes: dv.getUint32(8, true),
    layerCount: dv.getUint32(12, true),
    bins: dv.getUint32(16, true),
    flags: dv.getUint32(20, true),
    layers: [],
  };
  let off = 24;
  for (let i = 0; i < header.layerCount; i += 1) {
    header.layers.push({
      in: dv.getUint32(off, true),
      out: dv.getUint32(off + 4, true),
      min: dv.getFloat32(off + 8, true),
      max: dv.getFloat32(off + 12, true),
    });
    off += 16;
  }
  header.headerBytes = off;
  return header;
}

export function decodeHistogramRecord(dv, offset, header) {
  let off = offset;
  const step = dv.getUint32(off, true);
  off += 4;
  const layers = [];
  for (let l = 0; l < header.layerCount; l += 1) {
    const min = dv.getFloat32(off, true);
    const max = dv.getFloat32(off + 4, true);
    off += 8;
    const counts = new Float32Array(header.bins);
    for (let b = 0; b < header.bins; b += 1) {
      counts[b] = dv.getFloat32(off, true);
      off += 4;
    }
    layers.push({ min, max, counts });
  }
  return { step, layers };
}

// activations.bin 头：16 + 4 * 层数 字节
export function parseActivationHeader(buf) {
  if (buf.length < 16) throw new Error('activation header too short');
  const magic = readMagic(buf);
  if (magic !== 'TNA1') throw new Error(`bad activation magic: ${magic}`);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const header = {
    magic,
    version: dv.getUint32(4, true),
    recordBytes: dv.getUint32(8, true),
    layerCount: dv.getUint32(12, true),
    sizes: [],
  };
  let off = 16;
  for (let i = 0; i < header.layerCount; i += 1) {
    header.sizes.push(dv.getUint32(off, true));
    off += 4;
  }
  header.headerBytes = off;
  return header;
}

export function decodeActivationRecord(dv, offset, header) {
  let off = offset;
  const step = dv.getUint32(off, true);
  off += 4;
  const layers = [];
  for (let l = 0; l < header.layerCount; l += 1) {
    const size = header.sizes[l];
    const values = new Float32Array(size);
    for (let i = 0; i < size; i += 1) {
      values[i] = dv.getFloat32(off, true);
      off += 4;
    }
    layers.push(values);
  }
  return { step, layers };
}

// probe_inputs.bin：头 16 字节，随后 count * width 个 float32
export function parseProbeInputs(buf) {
  const magic = readMagic(buf);
  if (magic !== 'TNP1') throw new Error(`bad probe magic: ${magic}`);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const version = dv.getUint32(4, true);
  const count = dv.getUint32(8, true);
  const width = dv.getUint32(12, true);
  const pixels = [];
  let off = 16;
  for (let i = 0; i < count; i += 1) {
    const row = new Float32Array(width);
    for (let x = 0; x < width; x += 1) {
      row[x] = dv.getFloat32(off, true);
      off += 4;
    }
    pixels.push(row);
  }
  return { version, count, width, pixels, headerBytes: 16 };
}
