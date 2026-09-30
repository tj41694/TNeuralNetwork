# TNeuralNetwork — AGENTS.md

## 禁止事项
- 不要自动执行 `git commit`，只有用户明确要求时才提交。
- 含中文的注释/文档一律使用 UTF-8 编码。

## 构建与运行
CMake + Ninja + Clang（本机缓存为 LLVM `clang-cl`，Release）。无测试、无 CI。

```powershell
# 首次配置（生成 build/build.ninja）
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -B build
# 增量构建
cmake --build build          # 等价于 ninja -C build
```

运行必须在 `build/` 目录下执行——`main.cpp:9` 以相对路径 `resources/test.db` 打开数据库：

```powershell
cd build; .\NumDistinguish.exe
```

- `build/resources/test.db`（约 287MB）**不在 git 中**，新克隆后需自备；缺失或工作目录不对时程序直接 `return 1`。
- 训练循环固定 15000 轮、每 batch 打印一行（`NumDistinguish.cpp:60`），完整跑完耗时很长，不要误判为卡死。

## 代码约定
- `.clang-format`：LLVM 基础，Allman 花括号，4 空格缩进，列宽 100；**C 语言禁用格式化**（文件末尾 `Language: C / DisableFormat: true`）。不要格式化 `NumDistinguish/sqlite3/`。
- `.clang-tidy`：`Checks: '*'`。命名约定：类/结构体/枚举 `CamelCase`，函数/变量 `camelBack`，private/protected 成员后缀 `_`。
- clangd LSP 见 `opencode.json` + `.clangd`，依赖 `build/compile_commands.json`（CMake 已开启 `CMAKE_EXPORT_COMPILE_COMMANDS`）。改动源码后重建以刷新索引。

## 架构
单可执行文件，源码全在 `NumDistinguish/`：

| 文件 | 作用 |
|---|---|
| `main.cpp` | 入口：读 SQLite → 建 3 层网络 → `Training` → `Validate` |
| `NumDistinguish.h/.cpp` | `DigitalDistinguish` 模型（`PushLayer`/`Forward`/`Backward`/`UpdateWeights`）+ 激活函数 |
| `Sample.h/.cpp` | `TnVector`(vector<double>)、`TnLayer`(权重矩阵+偏置)、`Sample`(标签+数据)、`CostFunc` 枚举 |
| `Shuffle.h/.cpp` | mini-batch 随机采样池 |
| `sqlite3/` | 内嵌 SQLite 合并源，勿改 |

- 网络（`main.cpp:54-56`）：784 → 24(ReLU) → 16(ReLU) → 10(SoftMax)，损失 CrossEntropy，SGD lr=0.001、batch=100。
- **`TnLayer::values` 一值两用**：前向传播时是激活值，反向传播时被 `CalcGradient` 重写为梯度；同一对象在不同阶段含义不同，改代码时勿混用。
- 反向传播在 batch 内对共享 gradient layer 累加梯度，`UpdateWeights` 再除以 batchSize；gradient layer 由拷贝构造生成、零初始化。
- `matrix`/`bias` 为 public；`TnLayer::operator*=` 内完成矩阵乘 + 加 bias + 激活。
- 数据表 `tr_data`(训练) / `te_data`(测试)，列 `(id, label, image_blob)`；图像 784 个 float，仅当 blob 恰为 3136 字节时才加载。
- `Sigmoid` / `DerivSigmoid` / `DerivSoftMax` 仍是 TODO 空实现。
