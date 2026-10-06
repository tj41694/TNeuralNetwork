# TNeuralNetwork — AGENTS.md

- 无论什么情况都用中文对话与回答。

## 禁止事项
- 不要自动执行 `git commit`，只有用户明确要求时才提交。
- 含中文的注释/文档一律用 UTF-8。
- 不要格式化 `TNeuralNetworkEngine/sqlite3/` 与 `web/httplib/`（内置第三方源码）。

## 构建与运行
CMake + Ninja + clang/clang++（GNU 驱动，Release）。无测试、无 CI。

```powershell
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -B build  # 仅首次
cmake --build build                                                            # 增量
cd build; .\NumDistinguish.exe                                                 # 训练，必须在 build/ 下运行
cd build; .\NumDistinguishWeb.exe                                             # 只读遥测服务，同样在 build/ 下运行
```

构建产出**两个独立 exe**：`NumDistinguish`（只训练并写遥测）与 `NumDistinguishWeb`（只读服务，回看 `runs/`）；两者不共享内存，只通过文件通信。

- `main.cpp` 以相对路径打开 `resources/test.db`，所以工作目录必须是 `build/`。该库（约 287MB）不在 git 中，新克隆需自备，缺失直接 `return 1`。
- 训练常用参数：`--exp NAME`、`--seed N`、`--steps N`、`--probes N`、`--hist-range R`（一个值=所有层，或 `1,1.5,1.5,2` 按层给，这也是默认）、`--help`。
- Web 服务参数：`--out DIR`（默认 `../runs`）、`--web DIR`（默认 `../web`）、`--port N`（默认 5108，占用时自动换）、`--help`。
- 21000 步约 2 分钟（Release，本机），默认每 100 步打印一行；别因为打印稀疏就误判成卡死。

## 代码约定
- `.clang-format`：LLVM + Allman + 4 空格 + 列宽 100；C 语言禁用格式化。
- 命名：类型 `CamelCase`，函数/变量 `camelBack`，成员用 `m_` 前缀（`m_layers`、`m_realValue`）。`.clang-tidy` 里配的成员后缀 `_` 与现有代码不符，不要为迎合它改名。
- clangd 依赖 `build/compile_commands.json`；改动源码后重建以刷新索引。

## 架构
源码按职责分三个目录，功能代码与训练模型是解耦的：

| 目录 | 是什么 | 详见 |
|---|---|---|
| `TNeuralNetworkEngine/` | 模型本体与入口：`main.cpp`（CLI + 组装）、`NumDistinguish.*`（`DigitalDistinguish` + 激活函数）、`Sample.*`（`TnVector`/`TnLayer`/`Sample`/`CostFunc`）、`Shuffle.*`（mini-batch 采样池）、`TnRandom.*`（全局随机源）、`sqlite3/`（内嵌合并源，勿改） | 本文件 |
| `record/` | **训练遥测的写入端**：只负责把 loss/lr/范数、权重直方图、probe 激活写成 `runs/<exp>_<时间戳>/` 下的文件。不碰网络，不引用训练对象以外的任何东西 | [`record/README.md`](record/README.md) |
| `web/` | **整个 Web 功能**：独立入口 `main.cpp` + 只读 HTTP 服务（`DashboardServer.*` + `httplib/`，cpp-httplib v0.58.0）+ 纯静态前端（`index.html`/`app.js`/`protocol.js`/`charts.js`/`gl.js`/`style.css`）。编译成独立 exe `NumDistinguishWeb`，只读 `runs/`，不引用训练内存 | [`web/README.md`](web/README.md) |

**改动时的边界**：`record/` 只写文件、`web/` 只读文件，两者之间不共享内存也不需要锁；
文件格式是它们唯一的契约，改任何一方都要同步 `web/protocol.js`、`docs/telemetry-plan.md` §4 与 `tools/proto-check.mjs`。

模型细节（`TNeuralNetworkEngine/`）：

- 网络与超参（`main.cpp` 建网络、`TrainingOptions` 给默认值）：784 → 24(ReLU) → 24(ReLU) → 16(ReLU) → 10(SoftMax)，CrossEntropy，SGD，batch=100，默认 25000 步。
- 学习率在 `TrainingOptions` 里：step ≥ 17500 后由 0.1 降到 0.05。日志记录的是**真正传入** `UpdateWeights` 的值。
- 权重初始化在 `TnLayer` 构造函数里用 He/Kaiming：`N(0, sqrt(2/fan_in))`（`fan_in` 为该层输入维度），偏置置 0；随机源走 `TnRandom::RandomNormal`。
- **`TnLayer::values` 一值两用**：前向是激活值，反向被 `CalcGradient` 重写成梯度，勿混用。batch 内对共享 gradient layer 累加梯度（拷贝构造、零初始化）；反向一开始就把输出层梯度除以 batchSize，累加后即为 batch 平均梯度，`UpdateWeights` 只按学习率缩放。
- `matrix`/`bias`/`preActiveValues` 是 `protected`/`private`，但已有只读访问器 `Matrix()` / `Bias()` / `PreActiveValues()`；`operator*=` 内完成矩阵乘 + 加 bias + 激活。
- `DerivSoftMax` 的空实现是 softmax+CE 的**有意设计**（梯度已在 `Backward` 里写成 `output - onehot`），不要"补全"；`Sigmoid`/`DerivSigmoid` 才是真 TODO。
- 随机性统一走 `TnRandom`（mt19937 + 显式 seed 写进 `meta.json`），**不要**再用 `rand()` 或当前时间做种子，否则 run 之间无法对比。
- 数据表 `tr_data`(训练)/`te_data`(测试)，列 `(id, label, image_blob)`；图像 784 个 float，仅当 blob 恰为 3136 字节时才加载。

## 遥测与可视化
训练遥测写入**仓库根的 `runs/<exp>_<时间戳>/`**（默认 `--out ../runs`，已被 git 忽略），由独立的 `NumDistinguishWeb` 只读 HTTP 服务提供给浏览器；`runs/` 下每个文件由 `record/` 写入、由 `web/` 只读消费，训练结束后可单独运行 `NumDistinguishWeb` 继续回看。格式、协议与踩过的坑见 [`docs/telemetry-plan.md`](docs/telemetry-plan.md)。

前端是 `web/` 下的纯静态 ES module（无构建步骤），校验脚本在 `tools/`：

```powershell
node tools\proto-check.mjs <origin> <runName> <runsDir> [staticOrigin]   # 增量协议
node tools\charts-check.mjs <origin> <runName>                          # 图表数学
```
