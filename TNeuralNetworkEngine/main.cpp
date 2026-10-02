#include "DashboardServer.h"
#include "NumDistinguish.h"
#include "Recorder.h"
#include "Sample.h"
#include "TnRandom.h"
#include "sqlite3/sqlite3.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace
{
std::atomic<bool> g_stopRequested{false};

#if defined(_WIN32)
BOOL WINAPI ConsoleHandler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT)
    {
        // 只置标志位：控制台处理函数跑在另一个线程上，在这里做文件 I/O 有死锁风险.
        g_stopRequested.store(true);
        return TRUE;
    }
    return FALSE;
}

// 终端里中文乱码的根因：源码与 printf 走的是 UTF-8 字节，而 Windows 控制台默认按系统
// ANSI 代码页（简中为 936/GBK）解码这些字节，两者不一致就成了"鐢ㄦ硶"这种乱码。
// 把控制台输出代码页切到 UTF-8 让两边对齐；退出时恢复原值，免得影响同一窗口后续的程序。
UINT g_consoleCpToRestore = 0; // 0 表示没改过，不需要恢复

void RestoreConsoleOutputCp()
{
    if (g_consoleCpToRestore != 0)
    {
        SetConsoleOutputCP(g_consoleCpToRestore);
    }
}

void SetupConsoleOutputUtf8()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    // 输出被重定向到文件或管道时没有控制台代码页这回事：UTF-8 字节本来就是想要的编码.
    if (hOut == nullptr || hOut == INVALID_HANDLE_VALUE || !GetConsoleMode(hOut, &mode))
    {
        return;
    }

    const UINT previous = GetConsoleOutputCP();
    if (previous == 0 || previous == CP_UTF8 || !SetConsoleOutputCP(CP_UTF8))
    {
        return;
    }

    g_consoleCpToRestore = previous;
    std::atexit(RestoreConsoleOutputCp);
}
#endif

struct Options
{
    std::string expName = "exp";
    uint32_t seed = 0;
    bool seedGiven = false;
    uint32_t steps = 25000;
    int batchSize = 100;
    int port = 0;
    std::string outRoot = "../runs";
    std::string webRoot = "../web";
    bool serve = true;
    bool hold = true;
    bool serveOnly = false;
    uint32_t probes = 16;
    // 直方图固定分箱范围，一个值＝所有层，或按层各一个（逗号分隔）
    std::vector<double> histRanges{3.0};
};

void PrintUsage()
{
    printf("用法: NumDistinguish [选项]\n");
    printf("  --exp NAME       实验名，输出到 runs/<NAME>_<时间戳>/ （默认 exp）\n");
    printf("  --seed N         随机种子；省略则自动派生并记入 meta.json\n");
    printf("  --steps N        训练步数，1 步 = 1 个 batch （默认 25000）\n");
    printf("  --batch N        每批样本数 （默认 100）\n");
    printf("  --out DIR        遥测输出根目录 （默认 ../runs，即仓库根的 runs/）\n");
    printf("  --web DIR        前端静态文件目录 （默认 ../web）\n");
    printf("  --port N         HTTP 端口；0 表示自动选空闲端口 （默认 0）\n");
    printf("  --probes N       激活快照使用的固定样本数 （默认 16，0 表示关闭）\n");
    printf("  --hist-range R   直方图固定分箱范围 ±R；一个值用于所有层，"
           "或用逗号按层各给一个（如 3,3,3,6，默认 3）\n");
    printf("  --no-serve       不启动 HTTP 服务\n");
    printf("  --serve-only     不训练，只把 runs/ 用 HTTP 服务起来（回看历史 run）\n");
    printf("  --no-hold        训练结束后立即退出，不等回车\n");
}

bool ParseUint(const char *text, uint32_t &out)
{
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0'))
    {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}

bool ParseArgs(int argc, char **argv, Options &opt)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto requireValue = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                printf("参数 %s 缺少取值\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            PrintUsage();
            std::exit(0);
        }
        else if (arg == "--no-serve")
        {
            opt.serve = false;
        }
        else if (arg == "--no-hold")
        {
            opt.hold = false;
        }
        else if (arg == "--serve-only")
        {
            opt.serveOnly = true;
        }
        else if (arg == "--exp")
        {
            const char *v = requireValue("--exp");
            if (v == nullptr)
            {
                return false;
            }
            opt.expName = v;
        }
        else if (arg == "--out")
        {
            const char *v = requireValue("--out");
            if (v == nullptr)
            {
                return false;
            }
            opt.outRoot = v;
        }
        else if (arg == "--web")
        {
            const char *v = requireValue("--web");
            if (v == nullptr)
            {
                return false;
            }
            opt.webRoot = v;
        }
        else if (arg == "--seed")
        {
            const char *v = requireValue("--seed");
            uint32_t value = 0;
            if (v == nullptr || !ParseUint(v, value))
            {
                return false;
            }
            opt.seed = value;
            opt.seedGiven = true;
        }
        else if (arg == "--steps")
        {
            const char *v = requireValue("--steps");
            if (v == nullptr || !ParseUint(v, opt.steps))
            {
                return false;
            }
        }
        else if (arg == "--batch")
        {
            const char *v = requireValue("--batch");
            uint32_t value = 0;
            if (v == nullptr || !ParseUint(v, value))
            {
                return false;
            }
            opt.batchSize = static_cast<int>(value);
        }
        else if (arg == "--port")
        {
            const char *v = requireValue("--port");
            uint32_t value = 0;
            if (v == nullptr || !ParseUint(v, value))
            {
                return false;
            }
            opt.port = static_cast<int>(value);
        }
        else if (arg == "--probes")
        {
            const char *v = requireValue("--probes");
            if (v == nullptr || !ParseUint(v, opt.probes))
            {
                return false;
            }
        }
        else if (arg == "--hist-range")
        {
            const char *v = requireValue("--hist-range");
            if (v == nullptr)
            {
                return false;
            }
            // 一个值 = 所有层；逗号分隔 = 按层各一个
            const std::string spec = v;
            std::vector<double> ranges;
            size_t start = 0;
            while (start <= spec.size())
            {
                const size_t comma = spec.find(',', start);
                const size_t len = (comma == std::string::npos) ? std::string::npos : comma - start;
                const std::string part = spec.substr(start, len);
                char *end = nullptr;
                const double value = std::strtod(part.c_str(), &end);
                if (part.empty() || end == part.c_str() || *end != '\0' || !(value > 0.0))
                {
                    printf("--hist-range 取值非法: %s（应为正数，或用逗号分隔每层一个）\n",
                           spec.c_str());
                    return false;
                }
                ranges.push_back(value);
                if (comma == std::string::npos)
                {
                    break;
                }
                start = comma + 1;
            }
            opt.histRanges = ranges;
        }
        else
        {
            printf("未知参数: %s\n", arg.c_str());
            PrintUsage();
            return false;
        }
    }
    return true;
}

bool GetData(vector<Sample *> &datas, int type)
{
    sqlite3 *db = nullptr;
    if (SQLITE_OK != sqlite3_open("resources/test.db", &db))
    {
        printf("%s", sqlite3_errmsg(db));
        return false;
    }
    sqlite3_stmt *pStmt = nullptr;
    int prepareResult = SQLITE_ERROR;
    switch (type)
    {
    case 1:
        prepareResult = sqlite3_prepare(db, "select * from te_data", -1, &pStmt, 0);
        break;
    case 2:
    default:
        prepareResult = sqlite3_prepare(db, "select * from tr_data", -1, &pStmt, 0);
        break;
    }
    if (prepareResult != SQLITE_OK || pStmt == nullptr)
    {
        sqlite3_close(db);
        return false;
    }
    std::filesystem::create_directories("temp");
    int index = 0;
    while (sqlite3_step(pStmt) == SQLITE_ROW)
    {
        int ulImageSize = sqlite3_column_bytes(pStmt, 2);
        if (ulImageSize == 3136)
        {
            int realNum = sqlite3_column_int(pStmt, 1);
            Sample *layer = new Sample(realNum,
                                       (const float *) sqlite3_column_blob(pStmt, 2),
                                       ulImageSize / sizeof(float));
            datas.push_back(layer);
            index++;
        }
    }
    sqlite3_finalize(pStmt);
    sqlite3_close(db);
    return true;
}

// 训练结束后保持服务存活，方便继续看页面。
// 交互式终端按回车退出；非交互式（后台运行/管道）读不到输入就退化为一直存活，
// 靠 Ctrl+C 或外部终止结束——这样"训练跑完了但页面还在"才是可靠的行为。
void HoldServerAlive(const std::string &url)
{
    printf("\n仪表盘仍在 %s\n按回车退出（非交互式运行则保持存活，Ctrl+C 亦可）...\n", url.c_str());
    std::string line;
    std::getline(std::cin, line);
    if (!std::cin.eof())
    {
        return;
    }
    while (!g_stopRequested.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}
} // namespace

int main(int argc, char **argv)
{
#if defined(_WIN32)
    // 必须在任何 printf 之前：--help 与参数报错都要用中文.
    SetupConsoleOutputUtf8();
#endif

    Options opt;
    if (!ParseArgs(argc, argv, opt))
    {
        return 2;
    }
    if (opt.steps == 0)
    {
        opt.steps = 25000;
    }

#if defined(_WIN32)
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
#endif

    // 只看不练：把已有的 runs/ 服务起来，便于回看历史 run 与做端到端验证。
    if (opt.serveOnly)
    {
        DashboardServer server;
        std::string url;
        if (!server.Start(opt.outRoot, opt.webRoot, opt.port, url))
        {
            printf("HTTP 服务启动失败。\n");
            return 3;
        }
        printf("仪表盘: %s\n", url.c_str());
        HoldServerAlive(url);
        server.Stop();
        printf("done..\n");
        return 0;
    }

    if (opt.seedGiven)
    {
        SeedRandom(opt.seed);
    }
    const uint32_t seed = RandomSeed();

    vector<Sample *> datas, testDatas;
    if (!GetData(datas, 2) || !GetData(testDatas, 1))
        return 1;

    DigitalDistinguish model;
    model.PushLayer(784, 24, ReLU, DerivReLU);
    model.PushLayer(24, 24, ReLU, DerivReLU);
    model.PushLayer(24, 16, ReLU, DerivReLU);
    model.PushLayer(16, 10, SoftMax, DerivSoftMax);

    // 固定的 probe 集合：索引与标签写进 meta.json，像素写进 probe_inputs.bin，
    // 这样跨 step 的激活快照可比，页面上也能把输入图像与逐层激活并排画出来.
    vector<Sample *> probeSamples;
    vector<ProbeRef> probeRefs;
    vector<vector<double>> probePixels;
    const uint32_t probeCount =
        std::min<uint32_t>(opt.probes, static_cast<uint32_t>(testDatas.size()));
    for (uint32_t i = 0; i < probeCount; ++i)
    {
        probeSamples.push_back(testDatas[i]);
        ProbeRef ref;
        ref.index = i;
        ref.label = testDatas[i]->m_realValue;
        ref.source = "test";
        probeRefs.push_back(ref);
        probePixels.emplace_back(testDatas[i]->begin(), testDatas[i]->end());
    }

    RunMeta meta;
    meta.expName = opt.expName;
    meta.seed = seed;
    meta.datasetPath = "resources/test.db";
    meta.trainCount = datas.size();
    meta.testCount = testDatas.size();
    meta.costFunction = "CrossEntropy";
    meta.layers = {{784, 24, "ReLU"}, {24, 24, "ReLU"}, {24, 16, "ReLU"}, {16, 10, "SoftMax"}};
    meta.batchSize = static_cast<uint32_t>(opt.batchSize);
    meta.totalSteps = opt.steps;

    LoggingPolicy policy;
    // 一个值=所有层，填满=按层；其他个数视为写错，早失败好过默默取错范围
    if (opt.histRanges.size() != 1 && opt.histRanges.size() != meta.layers.size())
    {
        printf("--hist-range 给了 %zu 个值，但网络有 %zu 层；要么给 1 个（所有层通用），要么给 %zu 个\n",
               opt.histRanges.size(), meta.layers.size(), meta.layers.size());
        return 2;
    }
    policy.histRanges = opt.histRanges;

    Recorder recorder;
    const bool recording =
        recorder.BeginRun(opt.outRoot, meta, policy, probeRefs, probePixels, 784);
    if (recording)
    {
        printf("遥测目录: %s\n", recorder.RunDir().c_str());
    }
    else
    {
        printf("遥测目录创建失败，训练继续但不记录。\n");
    }

    std::string url;
    DashboardServer server;
    if (opt.serve)
    {
        if (!server.Start(opt.outRoot, opt.webRoot, opt.port, url))
        {
            printf("HTTP 服务启动失败（端口被占用？），训练继续。\n");
        }
        else
        {
            printf("仪表盘: %s\n", url.c_str());
            if (recording)
            {
                recorder.SetUrl(url);
            }
        }
    }

    TrainingOptions training;
    training.batchSize = opt.batchSize;
    training.totalSteps = opt.steps;
    training.recorder = recording ? &recorder : nullptr;
    training.probes = probeSamples.empty() ? nullptr : &probeSamples;
    training.stopRequested = &g_stopRequested;

    model.Training(datas, training);
    model.Validate(testDatas);

    if (recording)
    {
        recorder.EndRun(g_stopRequested.load() ? "interrupted" : "finished");
    }

    for (auto data : datas)
        delete data;
    for (auto data : testDatas)
        delete data;

    if (opt.hold && server.IsRunning())
    {
        HoldServerAlive(url);
    }

    server.Stop();
    printf("done..\n");
    return 0;
}
