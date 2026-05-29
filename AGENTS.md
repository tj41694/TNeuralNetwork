# TNeuralNetwork — AGENTS.md

## 构建

CMake + Ninja + Clang 跨平台构建。

```bash
# 配置（生成 build.ninja）
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -B build

# 编译
ninja -C build

# 运行（需在 resources/test.db 同级目录）
./build/NumDistinguish
```

最低要求：CMake ≥ 3.15，Ninja，Clang。

## 项目结构

| 路径 | 作用 |
|---|---|
| `NumDistinguish/main.cpp` | 入口——加载SQLite数据，创建3层网络，训练，测试 |
| `NumDistinguish/NumDistinguish.h/.cpp` | `DigitalDistinguish` 类——网络模型，前向/反向传播，SGD训练 |
| `NumDistinguish/NeuralMatrix.h/.cpp` | 权重矩阵，随机初始化为 [-1, 1] |
| `NumDistinguish/Sample.h/.cpp` | 训练样本，含激活层、代价函数（CrossEntropy / MeanSquare） |
| `NumDistinguish/Shuffle.h/.cpp` | mini-batch 索引洗牌器 |
| `NumDistinguish/sqlite3/` | 内嵌 SQLite 合并文件（无系统依赖） |

## 运行时

需要在当前工作目录下有 `resources/test.db`（SQLite）。表：
- `te_data` — 测试集（列：id, label, image_blob）
- `tr_data` — 训练集

图像 blob：784 个 float（28×28 灰度图），列类型 BLOB。

## 网络

前馈：784 → 24 → 16 → 10。隐藏层使用 Linear 激活，输出层使用 SoftMax。损失函数：CrossEntropy。优化器：SGD（lr=0.15, batch=100, 最多 5000 轮）。

## 备注

- 无测试、无 CI、无格式化/静态检查配置。
- 源码含中文注释。
- MIT 协议，作者 Jia Tang。
- `.gitignore` 为标准 Visual Studio 模板。
- **不要自动提交代码。** 只有用户明确要求时才执行 git commit。
