#pragma once
#include "Sample.h"
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

// 采样策略：会写进 meta.json，前端据此决定 x 轴与各面板的刷新频率.
struct LoggingPolicy
{
    uint32_t scalarsEvery = 1;
    uint32_t histEvery = 50;
    uint32_t actEvery = 200;
    uint32_t histBins = 64;
    // 直方图固定分箱范围 ±histRanges[l]，必须全 run 不变，否则热力图会横向漂移。
    // 只填一个值时对所有层生效；要按层区分就填满层数个。
    //
    // 默认 3 而不是 2 的来由（实测一次 25000 步的 run，用每层 counts 估分位数）：
    // 四层分布的 99% 质量都落在 ±1.1 ~ ±2.0 之内，而峰值分别是 1.69 / 1.70 / 2.20 / 4.47。
    // 取 ±3 让前三层的峰值完全不溢出、核心仍占 64 格中的 24~42 格；
    // 放宽到 ±5 只会把核心压到 14~25 格，白白损失分辨率。
    std::vector<double> histRanges{3.0};
    uint32_t heartbeatMs = 2000;
    uint32_t flushMs = 100;
    uint32_t printEvery = 100;

    double RangeFor(size_t layer) const
    {
        if (histRanges.empty())
        {
            return 3.0;
        }
        return histRanges[layer < histRanges.size() ? layer : histRanges.size() - 1];
    }
};

struct LayerDesc
{
    uint32_t input = 0;
    uint32_t output = 0;
    const char *activation = "";
};

struct ProbeRef
{
    uint32_t index = 0;
    int label = -1;
    const char *source = "test";
};

struct ScalarRecord
{
    uint32_t step = 0;
    uint64_t samplesSeen = 0;
    double elapsedMs = 0;
    int64_t wallMs = 0;
    double lr = 0;
    double loss = 0;
    double gradNorm = 0;
    double weightNorm = 0;
    double updateRatio = 0;
};

struct RunMeta
{
    std::string expName;
    uint32_t seed = 0;
    std::string datasetPath;
    uint64_t trainCount = 0;
    uint64_t testCount = 0;
    std::string costFunction = "CrossEntropy";
    std::vector<LayerDesc> layers;
    uint32_t batchSize = 100;
    uint32_t totalSteps = 25000;
    double lrHigh = 0.1;
    double lrLow = 0.05;
    uint32_t lrDecayFromStep = 20000;
};

// 训练遥测写入器。约定：只有训练线程调用它，HTTP 服务永远只读这些文件，
// 因此两者之间不需要任何锁.
class Recorder
{
  public:
    Recorder() = default;
    ~Recorder();
    Recorder(const Recorder &) = delete;
    Recorder &operator=(const Recorder &) = delete;

    bool BeginRun(const std::string &runsRoot, const RunMeta &meta, const LoggingPolicy &policy,
                  const std::vector<ProbeRef> &probes,
                  const std::vector<std::vector<double>> &probePixels, uint32_t probeWidth);

    void LogScalars(const ScalarRecord &record);
    void LogHistograms(uint32_t step, const std::vector<TnLayer *> &layers);
    void LogActivations(uint32_t step, uint32_t probeIndex, const std::vector<TnLayer *> &layers);
    void Heartbeat(uint32_t lastStep);
    void EndRun(const char *state, const std::string &error = std::string());

    void SetUrl(const std::string &url);

    const LoggingPolicy &Policy() const
    {
        return m_policy;
    }
    const std::string &RunName() const
    {
        return m_runName;
    }
    const std::string &RunDir() const
    {
        return m_runDir;
    }

  private:
    void FlushIfDue(bool force);
    void WriteStatus(uint32_t lastStep, const char *state, const std::string &error);
    void WriteProbeInputs(const std::vector<std::vector<double>> &probePixels, uint32_t probeWidth);
    void WriteMeta(const RunMeta &meta, const std::vector<ProbeRef> &probes);

    std::string m_runName;
    std::string m_runDir;
    std::string m_runId;
    std::string m_createdAt;
    std::string m_url;
    std::string m_state = "running";
    LoggingPolicy m_policy;
    RunMeta m_meta;
    FILE *m_scalars = nullptr;
    FILE *m_histograms = nullptr;
    FILE *m_activations = nullptr;
    std::vector<float> m_histCounts;
    uint64_t m_startMs = 0;
    uint64_t m_lastFlushMs = 0;
    uint64_t m_lastHeartbeatMs = 0;
    uint32_t m_lastStep = 0;
};
