# web/ —— 训练监控的整个 Web 功能

这个目录自洽地装下"浏览器看训练"所需的一切：**服务端 + 前端 + 第三方依赖**。
它只读 `runs/` 下的文件，不引用任何训练对象，所以和训练线程之间不需要锁。

完整规格（端点约定、增量协议、渲染取舍、坑清单）在 [`../docs/telemetry-plan.md`](../docs/telemetry-plan.md)；
这里只讲"这个目录是什么、改它要注意什么"。

## 目录布局

```
web/
  # 会通过 HTTP 提供给浏览器的静态资源
  index.html  style.css  app.js  protocol.js  charts.js  gl.js  package.json
  # 服务端（C++，不通过 HTTP 提供）
  DashboardServer.h  DashboardServer.cpp
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
- 端口传 0 让系统分配，实际 URL 会打印到控制台并写进 `status.json`。

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

训练跑完后服务不会退出，也可以完全不训练、只回看历史 run：

```powershell
cd build
.\NumDistinguish.exe --serve-only --port 5110 --out ../runs --web ../web
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

`protocol.js` 的任何解析改动都要和 `../record/README.md` 列的同一组文件保持一致
（`record/Recorder.*`、`docs/telemetry-plan.md` §4、`tools/proto-check.mjs`）。

## 已知未验证

WebGL2 热力图只经过代码审查，**没有在真实 GPU 上跑过**（开发环境无浏览器）；
2D 图表与协议逻辑都有 Node 侧的真实数据验证。
