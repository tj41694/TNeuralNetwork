#pragma once
#include <cstdint>
#include <string>
#include <vector>

// 命令行解析产物：所有可调项集中在这里，默认值与硬编码行为保持一致.
struct Options
{
    std::string expName = "exp";
    uint32_t seed = 0;
    bool seedGiven = false;
    uint32_t steps = 25000;
    int batchSize = 100;
    std::string outRoot = "../runs";
    uint32_t probes = 200;
    // 直方图固定分箱范围，一个值＝所有层，或按层各一个（逗号分隔）。
    // 默认按 He/Kaiming 初始化下的实测峰值给：0.85 / 1.37 / 1.34 / 1.86，各留约 10~20% 余量.
    std::vector<double> histRanges{1.0, 1.5, 1.5, 2.0};
};

// 解析命令行参数到 opt。--help 会打印用法并 exit(0)；
// 返回 false 表示参数有误（错误信息已打印到控制台）.
bool ParseArgs(int argc, char **argv, Options &opt);
