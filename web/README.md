# web/ —— 训练监控的整个 Web 功能

这个目录自洽地装下"浏览器看训练"所需的一切：**入口 + 服务端 + 前端 + 第三方依赖**。
它只读 `runs/` 下的文件，不引用任何训练对象，所以和训练之间不需要锁。
它编译成一个**独立于训练的 exe `NumDistinguishWeb`**（入口 `main.cpp`）：训练进程 `NumDistinguish`
只写 `runs/`，服务进程只读 `runs/`，两者不共享内存，只通过文件通信。

完整规格（端点约定、增量协议、渲染取舍、坑清单）在 [`../docs/telemetry-plan.md`](../docs/telemetry-plan.md)；
这里只讲"这个目录是什么、改它要注意什么"。

## 目录布局

```
web/
  # 会通过 HTTP 提供给浏览器的静态资源
  index.html  style.css  app.js  protocol.js  charts.js  gl.js  package.json
  # 入口与服务端（C++，不通过 HTTP 提供）
  main.cpp  DashboardServer.h  DashboardServer.cpp
  # 第三方依赖（不通过 HTTP 提供）
  httplib/httplib.h + LICENSE        cpp-httplib v0.58.0（MIT）
```

静态资源白名单只放行 `js / mjs / css / html / json / svg / png / ico` **且不含路径分隔符**，
所以 `DashboardServer.cpp`、`httplib/httplib.h`、本 README 都只会 404，不会被泄漏。
（`package.json` 会被提供，它只是给 Node/编辑器标记 ESM 的空壳。）

## 服务端要点

- 端点：`GET /` → `index.html`；`GET /<资源名>`；`GET /api/runs`；`GET /api/state?run=<name>`；
  `GET /runs/<name>/<file>`（带 Range）。非 GET/HEAD 一律 405，**全只读**。
- **只挂载 `/runs`，绝不能挂 `/`**：cpp-httplib 的内建文件服务在路由**之前**执行，挂 `/` 会吞掉 `/api/*`。
- Range 由 cpp-httplib 内建（206 / 416 / 后缀 / 多区间 / If-Range，且 206 自动禁用压缩），
  不要自己实现；但要在 `post_routing_handler` 里补 `Accept-Ranges`（库只在 HEAD 响应里补）
  和 `Cache-Control: no-store`（不禁缓存，浏览器的旧 body 会让增量 offset 算错）。
- 只绑 `127.0.0.1`：既避免 Windows 防火墙弹窗，也避免把 `runs/` 暴露到局域网。
- `run` 参数与资源名都做了字符白名单校验，且挂载点自身有规范化 + 前缀校验，防目录穿越。
- 端口传 0 让系统分配，实际 URL 会打印到控制台（服务是纯读者，不写 `status.json`）。

## 前端要点

- **无构建步骤**：原生 ES module，`index.html` 直接 `<script type="module">`。
  所以必须用 HTTP 打开（`file://` 会被模块的 CORS 规则拦住）。`package.json` 只是为了让
  Node 与编辑器按 ESM 解析 `.js`，方便跑测试脚本。
- **分工**：文字、坐标轴、折线走 2D canvas（`charts.js`）；只有直方图热力图走 WebGL2（`gl.js`）。
  **一个 WebGL2 context + 视口**，不要每个面板建一个（浏览器对 context 数量有上限，且丢失会互相牵连）。
- 热力图：纹理存**原始 counts**（`R32F`），全局尺度只改 `uLogMax` 这个 uniform，所以尺度变化不用重传纹理；
  每条新记录只 `texSubImage2D` 写一列。`R32F` 在 WebGL2 core 里**不可线性过滤**，用 `NEAREST`；
  上传前必须 `gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1)`。列数上限 8192（`histEvery=50` 时约 40 万步）。
- `protocol.js` 是纯逻辑、不依赖 DOM，Node 下可直接跑，协议错误基本都在这里。
- 轮询与渲染解耦：`setTimeout` 自调度（不要 `setInterval`，避免请求重叠），自适应间隔，
  页面隐藏时降频，`visibilitychange` 回来立刻追一次 —— 字节 offset 天然支持追赶。

## 开发与验证

服务独立成 exe，训练与否都不影响它；它只回看 `runs/`：

```powershell
cd build
.\NumDistinguishWeb.exe --port 5110 --out ../runs --web ../web
```

两个校验脚本（在 `../tools/`）：

```powershell
node ..\tools\proto-check.mjs http://127.0.0.1:5110 <runName> ..\runs [staticOrigin]
node ..\tools\charts-check.mjs http://127.0.0.1:5110 <runName>
```

- `proto-check.mjs` 用真实文件造半行 / 截断 / 换 run，覆盖 206、416、200 降级、尾部缓冲；
  传 `staticOrigin`（例如 `python -m http.server` 的地址，它**不支持 Range**）即可验证 200 降级路径。
- `charts-check.mjs` 用假 canvas 把真实数据喂给图表代码，断言所有绘制坐标有限，并覆盖空数据、单点、
  全常量、含 0/负值四个边界。

## 改协议时要同步的地方

`protocol.js` 的任何解析改动都要和 `../record/README.md` 列的同一组文件保持一致。
下面用**仓库根相对路径**列出这一组（仓库根 = 本目录的上一级）：`record/Recorder.*`、
`docs/telemetry-plan.md` §4、`tools/proto-check.mjs`。

## 面板怎么读

顶部标题栏显示所选 run 的状态、进度，以及训练结束后的最终测试集**准确率**（`status.json` 的 `accuracy`）；
run 下拉列表里也会为已评估的 run 标出 `acc x.x%`。

| 面板 | 画的是什么 |
|---|---|
| **Loss** | **蓝线** `#5ac8fa` 是该步 batch 的原始交叉熵（`scalars.jsonl` 的 `loss`，100 个样本的均值）；**黄线** `#ffd166` 是它的指数移动平均 |
| 学习率 | `lr`，真正传给 `UpdateWeights` 的值，能看到 step 17500 处的台阶（0.1 → 0.05） |
| 范数 | 红 `gradNorm` = 累加梯度的 L2（已在反向传播中除以 batchSize）；绿 `weightNorm` = 更新后权重与偏置的 L2 |
| 更新比 | `updateRatio = lr × gradNorm / weightNorm`，即这一步把权重挪动了多大比例 |
| 权重直方图 | 每 50 步一次的各层权重分布。横轴是采样序号，纵轴是 64 个分箱（下 = 小值），颜色是该 bin 的 counts 经对数归一化后的值。分箱范围**按层固定**在文件头里（默认 `1,1.5,1.5,2`，对应 He 初始化下的各层权重尺度），范围外的权重会被并进边缘 bin —— 标题右侧因此会报出实情："未截断（最大 \|w\| = L3 1.86）"、"轻微截断（L3 峰值 2.1>2；边缘 bin 占 1.8%，可忽略）"或"⚠ 截断明显（边缘 bin >5%）"。判定按**堆积质量**而不是"有没有权重越界"，因为输出层总有那么几个离群权重 |
| Probe 激活 | 每 200 步对 16 个固定测试样本采一次激活。左侧是输入图像，右侧每行一层的神经元格子，颜色按层内最大绝对值归一化 |

### Loss 面板的两条线

两条线是**同一份数据**，不是两个指标——黄色只是蓝色的平滑版本：

- **蓝线（原始）**：每一步单独一个 mini-batch（batch=100）的交叉熵均值。每步只见到 100 个样本，
  抖动极大：实测某个 run 的原始 loss 横跨 `[0.002, 2.3]`，仍跨约 3 个数量级。所以本面板用**对数 Y 轴**；
  并且当点数超过像素列的 2 倍时，蓝线按每个像素列取 **min–max** 抽稀、不透明度降到 0.55 ——
  看到的"宽带"就是抽稀的结果。带宽本身就是信息：越宽说明那个阶段 batch 间波动越大。
- **黄线（EMA）**：`acc = acc + 0.03 × (loss − acc)`，首步用第一个值初始化。
  有效时间常数约 `1 / 0.03 ≈ 33` 步，反映的是"最近三十来步的平均水平"。
  **判断有没有收敛要看黄线**，蓝带只用来看波动。
- 面板左上角图例会同时给出两个当前值，形如 `loss 0.155 · ema 0.189`。
- 开头几步 loss 约在 `ln(10) ≈ 2.30` 附近（实测 step 1 为 2.277）是正常的：权重用 He/Kaiming 初始化
  `N(0, sqrt(2/fan_in))`，初始输出接近均匀分布，交叉熵就接近随机猜测的 `-log(1/10)`。
  （旧版 `U(-1,1)` 初始化会让首层预激活值极大、loss 飙到几十，现已不再如此。）

想换平滑强度就改 `app.js` 里 loss 系列的 `emaAlpha`（当前 0.03）。

## 已知未验证

WebGL2 热力图只经过代码审查，**没有在真实 GPU 上跑过**（开发环境无浏览器）；
2D 图表与协议逻辑都有 Node 侧的真实数据验证。
