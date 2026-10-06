#include "CommandLine.h"
#include "NumDistinguish.h"
#include "Recorder.h"
#include "Sample.h"
#include "TnRandom.h"
#include "sqlite3/sqlite3.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <string>
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
#endif

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

int RunTraining(const Options &opt)
{
    if (opt.seedGiven)
    {
        SeedRandom(opt.seed);
    }
    const uint32_t seed = RandomSeed();

    vector<Sample *> datas, testDatas;
    if (!GetData(datas, 2) || !GetData(testDatas, 1))
        return 1;

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

    TrainingOptions training;
    training.batchSize = opt.batchSize;
    training.totalSteps = opt.steps;
    training.recorder = recording ? &recorder : nullptr;
    training.probes = probeSamples.empty() ? nullptr : &probeSamples;
    training.stopRequested = &g_stopRequested;

    DigitalDistinguish model;
    model.PushLayer(784, 24, ReLU, DerivReLU);
    model.PushLayer(24, 24, ReLU, DerivReLU);
    model.PushLayer(24, 16, ReLU, DerivReLU);
    model.PushLayer(16, 10, SoftMax, DerivSoftMax);
    model.Training(datas, training);
    model.Validate(testDatas);

    if (recording)
    {
        recorder.EndRun(g_stopRequested.load() ? "interrupted" : "finished");
    }

    for (auto *data : datas)
        delete data;
    for (auto *data : testDatas)
        delete data;

    printf("done..\n");
    return 0;
}
} // namespace

int main(int argc, char **argv)
{
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

    return RunTraining(opt);
}
