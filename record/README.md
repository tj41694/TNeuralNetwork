# record/ —— 训练遥测写入

这个目录只有一件事：**把训练过程中的数据可靠地写成文件**。它不碰网络，也不碰浏览器。

完整规格（字节布局、设计取舍、坑清单）在 [`../docs/telemetry-plan.md`](../docs/telemetry-plan.md)；
这里只讲"这个目录是什么、改它要注意什么"。

## 文件

| 文件 | 作用 |
|---|---|
| `Recorder.h` / `Recorder.cpp` | 唯一的写入器。`BeginRun` / `LogScalars` / `LogHistograms` / `LogActivations` / `Heartbeat` / `EndRun` |

## 输出（写到 `runs/<exp>_<时间戳>/`）

| 文件 | 形态 | 约 21000 步的体量 |
|---|---|---|
| `meta.json` | 一次性，含网络结构 / 超参 / seed / 采样策略 / probe 清单 | 1.4 KB |
| `status.json` | 持续覆盖更新：state、lastStep、heartbeat、url | 170 B |
| `scalars.jsonl` | 每步一行 JSON：step / loss / lr / gradNorm / weightNorm / updateRatio | 3.4 MB |
| `histograms.bin` | 定长记录：头 88 B + 每条 1060 B（step + 每层 min/max/64 个 counts） | 445 KB |
| `activations.bin` | 定长记录：头 32 B + 每条 300 B（step + 各层激活值） | 504 KB |
| `probe_inputs.bin` | 一次性：16 个 probe 样本的 784 个 float | 12.5 KB |

## 必须守住的不变量

1. **只有训练线程写**。HTTP 服务永远只读这些文件，所以两边不需要任何锁。
   要保住这条，改动时不要在服务端加写操作、也不要在 `Recorder` 里回调训练对象。
2. **要 `fflush`，不要 `fsync`**。服务和训练在同一台机器上共享 page cache，`fflush` 之后读者立刻可见；
   `FlushFileBuffers` 只影响掉电安全，这里是纯浪费。但**没有 `fflush` 页面就永远不动**。
   flush 按时间合并（默认 100 ms），并在 `EndRun` 与 Ctrl+C 路径强制刷一次。
3. **JSON 里绝不出现 `nan` / `inf` / `-nan(ind)`**。非有限值一律写 `null`（`FmtNumber` 负责），
   否则 `JSON.parse` 会抛异常、前端整条流都会受影响。
4. **`meta.json` / `status.json` 用"临时文件 + 覆盖式改名"原子替换**，避免读者读到半个 JSON。
   若目标正被 HTTP 服务以 mmap 打开（不带 `FILE_SHARE_DELETE`），改名会失败 —— 此时的兜底是就地写，
   内容最终正确、只留一个极小的撕裂窗口，前端要对 status 解析失败保持容忍。
5. **文件以允许并发读的方式打开**（Windows 上是 `_fsopen(..., _SH_DENYNO)`），
   否则服务端 mmap 时写入会拿到共享冲突。
6. **直方图分箱范围按层固定**（`LoggingPolicy::histRanges`，用 `RangeFor(l)` 取值），并写进文件头。
   只给一个值时所有层共用；要按层区分就填满层数个（`--hist-range 1,1.5,1.5,2`，也是默认）。
   每步各自算 min/max 会让热力图横向漂移，整个面板就废了。
   范围外的权重会被并进边缘 bin，所以每步的 min/max 必须照实记录 —— 前端靠它把"截断"标出来。
7. **定长记录**：`records = floor((文件大小 - 头长) / recordBytes)`。
   尾部不足一条的记录由读者留到下一轮拼接，写入方不需要做任何特殊处理。
8. 激活采样必须在 `UpdateWeights` **之后**、对固定 probe 样本单独跑一次前向，
   这样权重和激活对应同一个 step（见 `DigitalDistinguish::Training`）。

## 改格式时要同步的地方

动了任何字段名、记录布局或头结构，必须同时改这三处，否则会静默出错：

1. `../web/protocol.js` 里的解析器（`parseHistogramHeader` / `decodeActivationRecord` 等）
2. `../docs/telemetry-plan.md` 的 §4 格式表
3. `../tools/proto-check.mjs` 的断言（它会在真实文件上校验记录数与 step 间隔）

改完跑一遍：

```powershell
cd build; .\NumDistinguish.exe --exp fmt --steps 400      # 只训练并写遥测
cd build; .\NumDistinguishWeb.exe --port 5110             # 另开一个进程只读服务
node ..\tools\proto-check.mjs http://127.0.0.1:5110 <runName> ..\runs
```
