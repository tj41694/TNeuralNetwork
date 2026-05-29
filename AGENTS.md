# TNeuralNetwork — AGENTS.md

## 构建

仅限 Windows 的 Visual Studio 2019 (v142) C++17 项目。打开 `TNeuralNetwork.sln` 直接构建。无 CMake，不支持 Linux。

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
