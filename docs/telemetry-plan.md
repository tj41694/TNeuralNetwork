# 训练遥测与实时可视化方案

状态：**已实现并端到端验证**（2026-10-02）。
适用范围：`NumDistinguish` 单可执行文件 + 纯静态前端页面。

## 0. 实现状态

| 部分 | 位置 |
|---|---|
| 随机源（可复现） | `TNeuralNetworkEngine/TnRandom.h/.cpp` |
| 遥测写入 | `record/`（`Recorder.h/.cpp`，见 `record/README.md`） |
| 只读 HTTP 服务 | `web/`（`DashboardServer.h/.cpp` + `web/httplib/`，cpp-httplib v0.58.0，MIT） |
| 前端 | `web/`（protocol.js / charts.js / gl.js / app.js / index.html / style.css，无构建步骤） |
| 校验脚本 | `tools/proto-check.mjs`（增量协议）、`tools/charts-check.mjs`（图表数学） |

已验证的事实（不要凭直觉推翻）：

- `cpp-httplib` v0.58.0 **已内建**静态文件的 Range 支持：206 / 416 / 后缀范围 / 多区间 / If-Range 都正确，
  且 206 会自动走 identity 不走压缩。挂载点必须只挂 `/runs`——**内建文件服务在路由之前执行**，
  挂 `/` 会吞掉 `/api/*`。
- 它用句柄级 `GetFileSizeEx` 决定内容长度（不是可能陈旧的 NTFS 目录项），所以训练进程边写边读没问题；
  读取以 `FILE_SHARE_READ | FILE_SHARE_WRITE` 打开，与写入方的 `fopen(..., "ab")` 兼容。
- 训练 21000 步约 2 分钟（Release），最终测试集准确率 94.12%。
- `tools/proto-check.mjs` 23 项、`tools/charts-check.mjs` 13 项全部通过。

运行方式（三条命令的工作目录都是 `build/`，见 `AGENTS.md`）：

```powershell
.\NumDistinguish.exe --exp demo --steps 21000                 # 训练 + 仪表盘
.\NumDistinguish.exe --serve-only --port 5108                 # 不训练，只回看 runs/
node ..\tools\proto-check.mjs http://127.0.0.1:5108 <runName> ..\runs
```

## 1. 目标与非目标

**目标**

- 每次权重更新（**1 step = 1 个 batch**）落盘：`loss` / `lr` / `gradNorm` / `weightNorm`。
- 每 50 步：各层权重直方图。
- 每 200 步：固定 probe 样本的各层激活快照。
- 内嵌 HTTP 服务以**只读**方式把 `runs/` 提供给浏览器，支持 Range 增量读取。
- 浏览器纯静态页面增量拉取并渲染，**不做任何持久化**。

**非目标（本期不做）**

- 模型 checkpoint 的保存/加载。checkpoint 是给程序加载的，遥测是给人看的，两者的格式与生命周期不同，另行设计。
- 中断续跑训练。
- 训练算法本身的重构（唯一例外是 seed 可复现，见 §5.2）。

## 2. 架构约束（最重要的一条）

> **文件系统是训练进程与浏览器之间唯一的接口。**

由此推出三条硬约束，实现时必须守住：

1. `DashboardServer` 是**无状态纯读者**：只接收一个 `runs/` 根目录路径，不持有任何训练对象指针。
   → 训练线程与服务器线程之间**零锁**。
2. **只有训练线程写文件**，服务器永不写。
3. 训练结束后服务器可继续存活，页面仍能查看结果；历史 run 可以用任意静态服务器回看。

## 3. 目录布局

仓库根就是工作区根 `D:\projects\TNeuralNetwork\`；模型本体、遥测写入、Web 功能分在三个目录里。

```
D:\projects\TNeuralNetwork\            # 仓库根
  TNeuralNetworkEngine/                # 模型本体与入口
    main.cpp  NumDistinguish.*  Sample.*  Shuffle.*  TnRandom.*
    sqlite3/                           # 内嵌 SQLite 合并源，勿改
  record/                              # 遥测写入端（见 record/README.md）
    Recorder.h  Recorder.cpp
  web/                                 # 整个 Web 功能（见 web/README.md）
    index.html  style.css  app.js  protocol.js  charts.js  gl.js  package.json
    DashboardServer.h  DashboardServer.cpp
    httplib/                           # cpp-httplib v0.58.0 + LICENSE
  tools/                               # 校验脚本
    proto-check.mjs  charts-check.mjs
  docs/
    telemetry-plan.md                  # 本文件
  build/                               # 构建树（git 忽略；含 resources/test.db）
  runs/                                # 训练输出（git 忽略）
    <expName>_<YYYYmmdd-HHMMSS>/       # 目录名带时间戳，避免同名 run 互相覆盖
      meta.json                        # 一次性，原子替换写入
      status.json                      # 持续更新，原子替换写入
      scalars.jsonl                    # 追加，文本
      histograms.bin                   # 追加，定长记录
      activations.bin                  # 追加，定长记录
      probe_inputs.bin                 # 一次性，参与激活采样的样本像素
```

`runs/` 与 `build/` 都已加入 `.gitignore`。

## 4. 文件格式

### 4.1 通用约定

- 所有二进制 **little-endian**，紧密排布。前端一律用 `DataView` 按偏移读取并显式传 `littleEndian = true`，**不要假设结构体在内存中的布局**。
- 二进制文件全部使用**定长记录**：`记录号 = 字节偏移 / recordBytes`，这样增量读取时"尾部不足一条记录的部分留到下次"和文本文件用同一套逻辑。
- 文本文件 UTF-8 无 BOM。JSON 中**不允许出现 `nan` / `inf` / `-nan(ind)`**：非有限值必须写 `null`。
- JSON 的键统一用 `camelBack`（与工程的函数/变量命名一致）。

### 4.2 `meta.json`

一次写入后不再修改，用"临时文件 + rename"原子替换。前端靠它做**通用化渲染**，不要硬编码网络结构。

| 字段 | 说明 |
|---|---|
| `formatVersion` | 整数，前端遇到不认识的版本要明确报错而不是乱画 |
| `runId` | UUID，用于检测文件被截断/重写 |
| `createdAt` | ISO 8601 带时区 |
| `expName` | 实验名 |
| `seed` | 随机种子（见 §5.2） |
| `dataset` | `{path, trainCount, testCount}` |
| `network` | `{cost, layers:[{in, out, activation}]}` |
| `hyperparams` | `{batchSize, totalSteps, lrSchedule:[{fromStep, value}]}` |
| `logging` | `{scalarsEvery:1, histEvery:50, actEvery:200, histBins:64, histRange:[[min,max] × 层数]}` |
| `probes` | `[{index, label, source}]`，数组顺序即 `activations.bin` 内的 probe 顺序 |
| `scalarFields` | 每个标量的 `{key, label, unit, logScale}`，让前端不写死字段名 |

`histRange` 是**每层各自**的固定范围，**必须在这里定死、全 run 不变**（原因见 §9 坑 5）。
默认按层给 `1.0 / 1.5 / 1.5 / 2.0`，来由是 He/Kaiming 初始化下的一次 25000 步实测：
各层 `|w|` 峰值约为 `0.85 / 1.37 / 1.34 / 1.86`，默认值各留约 10~20% 余量，
核心质量能落在 64 格中较宽的区间；若沿用旧的单一 ±3，浅层分布会被压到中间极少分箱。
`--hist-range` 可以只给一个数（所有层通用）或按层给（如 `1,1.5,1.5,2`，也是默认值）。

**"截断"要按质量占比判断，不能按"有没有权重越界"**：判定阈值取"边缘 bin 质量 > 5%"
才算截断明显，其余只在 `tools/proto-check.mjs` 的 INFO 行和面板提示里给出实际峰值。

### 4.3 `status.json`

`meta.json` 写完后不应再被改写，而"是否还在训练"是持续变化的，因此单独放一个文件，每约 5 秒原子替换一次：

```json
{ "runId": "...", "state": "running|finished|crashed", "lastStep": 1234,
  "heartbeatMs": 1767000000000, "url": "http://127.0.0.1:8080/", "error": null }
```

前端判断"进程是否还活着"只看 `heartbeatMs` 的时效，不去猜文件 mtime。

### 4.4 `scalars.jsonl`

一行一个 JSON 对象，每步一行：

```json
{"step":1234,"samplesSeen":123400,"tMs":45231,"wallMs":1767000000000,
 "lr":0.1,"loss":0.4821,"gradNorm":1.234,"weightNorm":12.34,"updateRatio":0.00123}
```

指标定义（避免歧义，实现时按此计算）：

| 键 | 定义 | 计算时机 |
|---|---|---|
| `loss` | batch 内交叉熵均值，除以 `batchs.size()`（**不是**常量 `batchSize`） | `UpdateWeights` 之前 |
| `lr` | **真正传给 `UpdateWeights` 的值** | `UpdateWeights` 之前 |
| `gradNorm` | `sqrt(sum g²)`，g 为累加后的梯度（已在反向传播中除以 batchSize） | `UpdateWeights` 之前 |
| `weightNorm` | `sqrt(sum w²)`，含 `matrix` 与 `bias` | `UpdateWeights` 之后 |
| `updateRatio` | `lr * gradNorm / weightNorm`，即这一步把权重挪动了多大比例 | `UpdateWeights` 之后 |

这五个量只需对约 2 万个 double 扫两遍，成本可忽略，但 `gradNorm` / `weightNorm` / `updateRatio` 对诊断发散远比 `loss` 直接，**第一版就要带上**。

- 每层单独的范数可作为可选的 `perLayer` 数组后续再加。
- 数字用 `%.6g` 之类的短表示；25000 行约 2.7 MB，完全可接受。
- 不要用 CSV：JSONL 加字段不破坏旧解析器，CSV 加列会让位置解析全崩。

### 4.5 `histograms.bin`

头（`24 + 16 × L` 字节，L = 层数，4 层时 88 字节）：

| 偏移 | 类型 | 内容 |
|---|---|---|
| 0 | char[4] | `"TNH1"` |
| 4 | uint32 | `formatVersion` = 1 |
| 8 | uint32 | `recordBytes` |
| 12 | uint32 | `layerCount` |
| 16 | uint32 | `bins` |
| 20 | uint32 | `flags`（保留，0） |
| 24 | × L | 每层：`uint32 in`, `uint32 out`, `float32 binMin`, `float32 binMax` |

记录（定长，`recordBytes = 4 + L × (2 + bins) × 4`，bins=64、L=4 时 **1060 字节**）：

| 偏移 | 类型 | 内容 |
|---|---|---|
| 0 | uint32 | `step` |
| 4 | × L | 每层：`float32 min`, `float32 max`, `float32 counts[bins]` |

- `counts` 用 `float32`；要省空间可以归一化成 `uint8`，但**归一化基准也必须固定**。
- `min`/`max` 是当前 step 的实际权重范围，只用于画曲线和检测饱和，**不参与分箱**。
- 尺寸：每 50 步一条 → 500 条 × 1060 B ≈ 530 KB。

### 4.6 `activations.bin` 与 `probe_inputs.bin`

**不要 `<step>.bin` 一文件一步**（原因见 §9 坑 6）。隐藏层只有 24/24/16/10 个神经元，整条记录小到可以忽略，用单文件定长记录。

`activations.bin` 头（`16 + 4 × L` 字节，4 层时 32 字节）：`"TNA1"`(4) + `formatVersion`(4) + `recordBytes`(4) + `layerCount`(4) + 每层 `uint32 size`。

记录（定长，`recordBytes = 4 + sum(size) × 4`，4 层时 **300 字节**）：`uint32 step` + 各层 `float32 values[size]`（`Values()` 里**激活后**的值）。

- **一条记录 = 一个 (step, probe) 对**，先按 step、再按 `meta.probes` 的顺序排列。
- 尺寸：每 200 步 × 16 个 probe × 300 B ≈ 600 KB。

`probe_inputs.bin` 头 16 字节：`"TNP1"`(4) + `formatVersion`(4) + `count`(4) + `width`(4)，随后 `count × width` 个 `float32`，行主序。**标签与索引放在 `meta.json` 的 `probes` 里**（文本、可读），二进制里只放像素。

单次 run 总占用 < 4 MB。

## 5. C++ 侧改动清单

分三层，**不要在训练循环里散落 `fopen`**。

### 5.1 访问器（前置条件）

`TnLayer`（`TNeuralNetworkEngine/Sample.h`）的 `matrix` / `bias` 是 `protected`，`preActiveValues` 是 `private`，**都没有访问器**，现在根本读不到权重。需要新增只读访问器：

- `const vector<vector<double>> &Matrix() const`
- `const vector<double> &Bias() const`
- `const TnVector &PreActiveValues() const`（记录激活前 z 时需要，可后置）

### 5.2 随机种子可复现

- `TnLayer` 构造函数里用 `rand()` + `srand(time(0))`（`TNeuralNetworkEngine/Sample.cpp`），`Shuffle` 构造函数里用 `chrono::system_clock::now()` 做种子（`TNeuralNetworkEngine/Shuffle.cpp`）。**两处都不可复现**，只换其中一处没有意义。
- 统一改用一个由外部传入的 `std::mt19937`，seed 由命令行/配置提供，并写进 `meta.json`。
- 否则页面上的多 run 对比毫无意义。

### 5.3 `Recorder`（新类，`record/Recorder.h/.cpp`）

`BeginRun(meta)` / `LogScalars(...)` / `LogHistograms(...)` / `LogActivations(...)` / `EndRun(state)`，内部持有 `FILE*`、预分配的直方图缓冲、待写缓冲。要点：

- **需要 `fflush`，不需要 `fsync`。** 服务器和训练在同一台机器上，共享同一份 page cache，`fflush` 之后读者立刻可见；`FlushFileBuffers` / `_commit` 只影响掉电安全，对本需求是纯浪费。
- 但**没有 `fflush` 就什么都看不到**——这是最容易"代码没错但页面不动"的地方。
- flush 策略：**按时间合并**（最多每 100 ms 一次），而不是每步都 flush；在 `EndRun` 和 `SetConsoleCtrlHandler`（Ctrl+C）里强制 flush，否则中断会丢尾部数据。
- 所有数值过一遍 `std::isfinite`，非有限写 `null`。
- `meta.json` / `status.json` 用临时文件 + rename 原子替换。
- Windows 共享模式：MSVC 的 `fopen`/`fstream` 默认共享打开，通常允许并发读；若改用 `CreateFileW`，必须带 `FILE_SHARE_READ | FILE_SHARE_WRITE`。

### 5.4 训练循环的插入点

- 每步 1 次 `LogScalars`；`step % 50 == 0` 时 `LogHistograms`；`step % 200 == 0` 时 `LogActivations`。
- **probe 采样必须在 `UpdateWeights` 之后**：对 `meta.probes` 里的每个样本单独跑一次 `Forward`，再读各层 `Values()`。这样权重与激活对应同一个 step。
  - 额外调 `Forward` 会覆盖 `m_layers[i]->values`，这是无害的（下一轮会完整重算），但**绝不能放进 batch 循环内部**，也不要碰 gradient layers。
- 修正现有的 lr 打印错误（见 `AGENTS.md`）：`printf` 现在打印的是常量 `lRate`，要打印真正传入 `UpdateWeights` 的值，日志更不能照抄。
- 现有每步 `printf` 在 Windows 控制台是加锁文本 I/O，日志上线后建议降频（如每 100 步），避免监控反而拖慢训练。

### 5.5 `DashboardServer`（新类）

见 §6。用 cpp-httplib（单头文件、MIT，与工程"内嵌 sqlite3 amalgamation"的风格一致）跑在 `std::thread` 里，训练结束时 `stop()` + `join()`。

### 5.6 `CMakeLists.txt`

新增 `record/Recorder.cpp`、`web/DashboardServer.cpp`，把 `TNeuralNetworkEngine/`、`record/`、`web/` 加进 include 路径，并 `target_link_libraries(NumDistinguish PRIVATE ws2_32)`。

> 当前工具链是 **GNU 风格的 `clang++` 驱动**（不是 `clang-cl`），因此**不要依赖 `#pragma comment(lib, ...)`**，必须在 CMake 里显式链接。源码里 `<winsock2.h>` 必须早于 `<windows.h>`，并定义 `WIN32_LEAN_AND_MEAN` / `NOMINMAX`，别忘了 `WSAStartup`。

## 6. HTTP 服务约定

### 6.1 端点

| 端点 | 说明 |
|---|---|
| `GET /` | `web/index.html` |
| `GET /app.js` `/gl.js` `/style.css` | `web/*` |
| `GET /api/runs` | `{"runs":[{name, createdAt, state, lastStep, bytes}]}` |
| `GET /api/state?run=<name>` | `{"meta":{...}, "status":{...}, "files":{"scalars.jsonl":{"size":N}, ...}}` |
| `GET /runs/<name>/<file>` | 静态文件，支持 Range |
| 其它方法 | `405`，**全部只读 GET** |

前端靠 `/api/state` 发现文件与长度，**不要让它猜文件名**。

### 6.2 响应头

- 所有文件响应都带 `Accept-Ranges: bytes`（前端的回退判断依赖它，**200 响应也要带**）。
- `Cache-Control: no-store`。不做这个，浏览器可能返回缓存的旧 body，增量逻辑会永久卡死。
- `Content-Type`：`.jsonl` → `text/plain; charset=utf-8`，`.json` → `application/json; charset=utf-8`，`.bin` → `application/octet-stream`。
- **禁用 gzip**：Range 的字节偏移是相对**编码后**表示的，一旦压缩偏移全错。

### 6.3 Range 语义

| 请求 | 响应 |
|---|---|
| 无 `Range` | `200` + 全量 |
| `bytes=N-`，`N < size` | `206` + `Content-Range: bytes N-(size-1)/size` |
| `bytes=N-`，`N == size` | **`416`**（按规范起点等于当前长度即不可满足）。前端要当成"暂无新数据"，不是错误 |
| `bytes=-N`（后缀） | `206`，取最后 N 字节。首屏"只拉最后 M 条记录"靠它，但前端不能假设它一定被支持 |
| 多区间 `bytes=0-10,20-30` | 不支持：忽略 Range 返回 `200`。前端也不要用 |

- 发送时按**请求开始那一刻快照的长度**读取，`Content-Length` 必须等于实际发送的字节数（文件在发送期间继续增长是正常的）。
- 服务端**不需要自己实现 Range**：cpp-httplib v0.58.0 已经做对了 206 / 416 / 后缀范围 / 多区间 / If-Range，
  并且 206 自动禁用内容压缩（压缩会让字节偏移指向编码后的表示）。用 `set_mount_point` 即可，
  但要记得禁止缓存、并且在 `post_routing_handler` 里补 `Accept-Ranges`（库只在 HEAD 响应里补）。

### 6.4 安全与运行

- 只绑 `127.0.0.1`：既避免 Windows 防火墙弹窗（绑 `0.0.0.0` 才弹），也避免把 `runs/` 暴露到局域网。
- 路径穿越：把请求路径 `weakly_canonical` 后校验前缀仍在 `runs/` 之内；拒绝 `..`、绝对路径、Windows ADS（`文件:流`）。
- 端口占用则顺延，最终 URL 打印到控制台并写进 `status.json`。
- keep-alive：浏览器会复用连接。用 cpp-httplib 无需操心；若手写 Winsock，要么正确处理 keep-alive 循环，要么每个响应都带 `Connection: close`。
- 训练结束后**让服务继续活着**（打印"打开 http://…/ 查看结果，按回车退出"），否则页面会突然失联。

## 7. 前端增量拉取协议

### 7.1 启动

1. `GET /api/state` 拿 `meta` + 各文件 `size` + `status`。
2. 文本文件（`scalars.jsonl`）全量拉取（约 2.7 MB，本地毫秒级）。
3. 二进制文件先拉头部解析 `recordBytes`，再拉**最后 M 条记录**：`bytes = size - M × recordBytes`。若服务器返回 `416` 或 `200`，退回全量拉取。
4. 进入轮询。

### 7.2 每轮状态机

```
GET /runs/<run>/<file>   Range: bytes=<offset>-
  206 -> 追加 body；offset += body.length
  200 -> 服务器忽略了 Range，body 从 0 开始：
           若 body.length < offset   -> 文件被截断/重写 -> 全量重置
           否则 body = body[offset:] -> 追加           <- 关键回退路径
  416 -> 没有新数据（文件长度没变）
  404 -> 文件还没创建，继续等
  5xx / 网络错 -> 退避重试 1s → 2s → 5s，界面显示"离线"

文本文件:
  pending += 新字节
  parts   = pending.split('\n')
  pending = parts.pop()                 # 最后一段可能不完整，留到下一轮
  逐行 JSON.parse；失败的行跳过 + 计数 + 界面显示警告徽标
  pending 超过上限（如 1MB）-> 判定为损坏，重置到文件末尾

二进制文件:
  n       = floor(pending.length / recordBytes) * recordBytes
  pending = pending[n:]
```

### 7.3 调度与其它要求

- 用 `setTimeout` 自调度，**不要 `setInterval`**（避免请求重叠）。
- 自适应间隔：有新数据 250 ms，空闲 1~2 s。
- 页面隐藏时降频到 5~10 s：后台标签页的定时器会被节流到 1 s 甚至 1 min，且 `requestAnimationFrame` 完全停止。`visibilitychange` 回到前台时立刻拉一次，并把这一大段 delta 一次性吸收——**字节 offset 天然支持追赶**，这是这个协议最好的性质。
- `fetch(url, {cache: 'no-store'})`：服务端和客户端两头都要禁用缓存。
- **"服务器不支持 Range"必须是可工作的降级路径，而不是错误路径**：`fetch` 用 200 回退。这带来一个额外好处——历史 run 可以直接用 `python -m http.server` 查看（它不支持 Range，会返回 200，正好当回退逻辑的测试用例）。

### 7.4 前缀稳定性

append-only 的文件，任何时刻 `[0, size)` 的内容都不变，所以 offset 语义永远安全。**唯一破坏它的是截断/重写**，因此需要 `runId` 校验 + size 收缩检测双保险。

## 8. 渲染分工

### 8.1 2D canvas 与 DOM（默认选择）

- **坐标轴、刻度、图例、文字一律不用 WebGL 画**，用 DOM 或叠一层 2D canvas。否则要实现 SDF 字体图集，纯属浪费时间。
- **折线面板（loss / lr / gradNorm / weightNorm）默认用 2D canvas**：代码量小、文字方便、25000 个点毫无压力。
- WebGL2 用在：点数极大（阈值约 20 万点）、直方图热力图、激活热力图，以及后续的 3D 展示。

### 8.2 WebGL2 注意事项

- **一个 WebGL2 context + 多视口（`viewport` + `scissor`）**，不要每个面板一个 canvas：浏览器对同时存在的 WebGL context 数量有上限（通常 8~16），且上下文丢失会互相牵连。
- 直方图热力图：一张 `宽 × bins` 的纹理，每次轮询只 `texSubImage2D` **一列**；用全屏四边形在 fragment shader 里做 colormap。
- 纹理格式：`R32F` 在 WebGL2 core 中**不可线性过滤**（需要 `OES_texture_float_linear`），用 `R8`/`R16F` 配 `NEAREST` 更省心。
- 上传前 `gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)`：列宽不是 4 的倍数时（单列上传、64 bin 的 uint8 等）不设这一句必然错位。
- counts 先做 `log`/`sqrt` 压缩再归一化，否则早期的大尖峰会把后期细节全压平。
- 折线用单 VBO + `LINE_STRIP` + 一次 draw call，增量数据用 `bufferSubData` 追加；点数超过画布宽度时按像素列取 min/max 抽稀。
- 渲染用脏标记 + `requestAnimationFrame`（数据没变不重绘），轮询用 `setTimeout`，两者解耦。
- 必须处理 `webglcontextlost`（阻止默认行为并重建 GPU 资源）；`canvas.width = cssWidth × devicePixelRatio`；用 `ResizeObserver` 响应尺寸变化。
- 二进制解析统一 `DataView` + `littleEndian = true`。

### 8.3 图表可用性

- loss 面板用**对数轴**。
- 每步只有 100 个样本，原始 batch loss 抖动极大：**必须加一条 EMA/移动平均曲线**，否则看不到趋势。
- Y 轴 autoscale 要带迟滞（只在超出当前范围一定比例时扩展，收缩平滑或干脆不自动收缩），并提供"手动锁定"，否则每帧抖动。

## 9. 坑清单

| # | 坑 | 后果 | 对策 |
|---|---|---|---|
| 1 | **NaN/Inf 写进 JSONL** | `printf("%g", NaN)` 输出 `nan` / `-nan(ind)`，`JSON.parse` 直接抛异常，一行坏数据可能掐断整条流。`SoftMax` 没做减最大值，NaN 几乎必然出现 | C++ 侧 `isfinite` 检查写 `null`；前端解析失败跳过该行并告警 |
| 2 | 服务器忽略 Range 返回 200 | 朴素客户端把整个文件当新增数据，数据重复 | 检测 206/200，200 时按自身 offset 切分 body |
| 3 | 把 416 当错误 | 更隐蔽的是：**文件被截断后请求也会命中 416**（起点 ≥ 新长度），若只当成"没有新数据"，增量同步会永远卡住，且再也恢复不了 | 416 时额外发一个 `Range: bytes=0-0` 问出真实长度（该响应不带 Content-Range）：长度 < 自身 offset 即判定截断并整体重置。空闲时把这次探测限流到每 2 秒最多一次 |
| 4 | 文件被截断/同名 run 覆盖 | offset 跑到文件末尾之外，永远拉不到数据 | 目录名带时间戳；`runId` + size 收缩 → 全量重置 |
| 5 | **直方图 bin 范围每步重算** | 热力图横向漂移，整个面板失去意义 | 范围固定在文件头，全 run 不变 |
| 6 | `activations/<step>.bin` 一文件一步 | 25000 个小文件：目录枚举慢、Defender 逐个扫描拖慢训练、前端还得先发现文件名 | 单文件 + 定长记录（隐藏层总共 74 个 float） |
| 7 | `matrix`/`bias`/`preActiveValues` 无访问器 | 根本写不出权重直方图 | 先加只读访问器（`AGENTS.md` 曾误写为 public） |
| 8 | `values` 一值两用 + 采样时机 | 采到 batch 里最后一个样本，或与权重不对应同一 step | `UpdateWeights` 之后对固定 probe 跑一次 `Forward` |
| 9 | 没有 `fflush` | 代码全对，页面就是不动 | 按时间合并 flush；Ctrl+C 处理里强制 flush |
| 10 | 浏览器/服务端缓存 | 拿到旧 body，增量逻辑永久卡死 | `cache:'no-store'` + `Cache-Control: no-store` |
| 11 | 多 WebGL context / R32F 过滤 / `UNPACK_ALIGNMENT` | 面板黑屏、热力图错位 | 单 context 多视口；R8/R16F + NEAREST；上传前设 alignment |
| 12 | 后台标签页节流 | 切回来时画面冻住 | 字节 offset 支持追赶；`visibilitychange` 立即刷新 |
| 13 | 路径穿越 / 绑 `0.0.0.0` | 把磁盘暴露给局域网 | 只绑 loopback，canonical 校验前缀，只允许 GET |
| 14 | keep-alive 处理不当（手写服务器） | 第一个请求正常，后续请求挂死 | 用 cpp-httplib，或每响应带 `Connection: close` |
| 15 | 每步 printf + 每步 I/O | 监控反而拖慢训练 | 先测基线；printf 降频；复用缓冲避免每步分配 |
| 16 | `rand()` / `chrono` 做种子 | 多 run 无法对比 | 统一 `mt19937` + 显式 seed，写进 `meta.json` |
| 17 | `runs/` 无限增长 | 日志吃满磁盘 | 加保留策略（最多 N 个 run / 总量上限），并加入 `.gitignore` |
| 18 | lr 打印的是常量 0.1 | 页面上学习率曲线是假的，掩盖衰减 | 记录真正传入 `UpdateWeights` 的值 |

## 10. 里程碑

| 阶段 | 内容 |
|---|---|
| **M0** | 定死格式；写一个**假日志生成器**，按真实节奏往 `runs/demo/` 追加，并能主动制造半行、NaN、截断、重启。前端不等训练，增量协议和降级逻辑都靠它测 |
| M1 | `TnLayer` 访问器 + seed 改造 + `Recorder`：只落地 `meta.json` / `status.json` / `scalars.jsonl` |
| M2 | 前端首屏全量拉取 + 轮询；先用 `python -m http.server`（走 200 回退路径）验证 |
| M3 | `DashboardServer` + `/api/state` + Range/206/416 |
| M4 | 折线面板（loss 对数轴 + EMA + lr/gradNorm/weightNorm） |
| M5 | `histograms.bin` + 热力图面板 |
| M6 | probe 样本 + `activations.bin` + 输入图像/激活可视化 |
| M7 | 可选：per-class accuracy、混淆矩阵、3D 展示 |

M0–M6 均已完成并通过验证（见 §0）。其中 M0 改了做法：没有写假日志生成器，改用
`tools/proto-check.mjs` 在**真实文件**上注入半行 / 截断 / 换 run，并用 `python -m http.server`
充当"忽略 Range 的服务器"，比造假数据更接近真实故障。M7 与 checkpoint 属于独立增量。


## 11. 未决 / 后续

- **accuracy / 混淆矩阵**：需要周期性评估（插在 `UpdateWeights` 之后是安全的，不会破坏梯度累加），目前不在每步标量里。建议放 `metrics.jsonl`，低频。**注意 `Validate()` 目前只在训练结束后跑一次，所以页面上没有准确率曲线。**
- **激活前 z**：访问器已经加上（`TnLayer::PreActiveValues()`），但记录格式里还没用。
- **checkpoint 格式与生命周期**：独立设计，不与遥测混用。
- **权重轨迹采样**：在固定随机索引上记录一小撮权重随 step 的变化，比直方图更适合做 3D/轨迹展示，成本极低。
- **`runs/` 保留策略**：尚未实现（单个 21000 步 run 约 4.2MB）。
- **热力图列数上限 8192**：即 `histEvery=50` 时约 40 万步，超过后新记录被丢弃并在页面上提示。
- **浏览器端没有自动化验证**：WebGL2 面板只经过代码审查，未在真实 GPU 上跑过；Node 侧只覆盖了格式解析与 2D 图表数学。

