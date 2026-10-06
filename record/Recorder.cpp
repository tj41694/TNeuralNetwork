#include "Recorder.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <share.h>
#endif

namespace
{
uint64_t NowSteadyMs()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

int64_t NowWallMs()
{
    using namespace std::chrono;
    return static_cast<int64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

void LocalAndUtc(std::time_t t, std::tm &local, std::tm &utc)
{
#if defined(_WIN32)
    localtime_s(&local, &t);
    gmtime_s(&utc, &t);
#else
    localtime_r(&t, &local);
    gmtime_r(&t, &utc);
#endif
}

std::string IsoNow()
{
    std::time_t t = std::time(nullptr);
    std::tm local{};
    std::tm utc{};
    LocalAndUtc(t, local, utc);

    char body[32] = {0};
    std::strftime(body, sizeof(body), "%Y-%m-%dT%H:%M:%S", &local);

    char zone[16] = {0};
    std::strftime(zone, sizeof(zone), "%z", &local);

    std::string offset;
    if (std::strlen(zone) == 5) // +0800 -> +08:00
    {
        offset = std::string(zone, 3) + ":" + std::string(zone + 3, 2);
    }
    return std::string(body) + offset;
}

std::string TimestampForDir()
{
    std::time_t t = std::time(nullptr);
    std::tm local{};
    std::tm utc{};
    LocalAndUtc(t, local, utc);
    char buf[32] = {0};
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &local);
    return buf;
}

std::string MakeRunId()
{
    static const char *hex = "0123456789abcdef";
    std::random_device rd;
    std::string id;
    id.reserve(32);
    for (int i = 0; i < 32; ++i)
    {
        id.push_back(hex[rd() & 0xF]);
    }
    return id;
}

std::string JsonEscape(const std::string &s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            }
            else
            {
                out.push_back(static_cast<char>(c));
            }
            break;
        }
    }
    return out;
}

// JSON 里不允许出现 nan / inf / -nan(ind)：非有限值一律写 null.
std::string FmtNumber(double v)
{
    if (!std::isfinite(v))
    {
        return "null";
    }
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.6g", v);
    return buf;
}

uint32_t Magic(char a, char b, char c, char d)
{
    return static_cast<uint32_t>(static_cast<unsigned char>(a)) |
           (static_cast<uint32_t>(static_cast<unsigned char>(b)) << 8) |
           (static_cast<uint32_t>(static_cast<unsigned char>(c)) << 16) |
           (static_cast<uint32_t>(static_cast<unsigned char>(d)) << 24);
}

// 以允许并发读的方式打开：训练进程在追加写，HTTP 服务随时可能在读同一个文件.
FILE *OpenSharedWrite(const std::string &path)
{
#if defined(_WIN32)
    return _fsopen(path.c_str(), "wb", _SH_DENYNO);
#else
    return std::fopen(path.c_str(), "wb");
#endif
}

void WriteU32(FILE *fp, uint32_t v)
{
    unsigned char b[4] = {static_cast<unsigned char>(v & 0xFF),
                          static_cast<unsigned char>((v >> 8) & 0xFF),
                          static_cast<unsigned char>((v >> 16) & 0xFF),
                          static_cast<unsigned char>((v >> 24) & 0xFF)};
    std::fwrite(b, 1, sizeof(b), fp);
}

void WriteF32(FILE *fp, float v)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    WriteU32(fp, bits);
}

// 先写临时文件再覆盖式改名，避免读者读到半个 JSON。
// 若目标正被 HTTP 服务以 mmap 打开（不带 FILE_SHARE_DELETE），改名会失败，
// 此时退化为就地写：内容最终仍然正确，代价是极小的撕裂窗口。
bool WriteFileAtomic(const std::string &path, const std::string &content)
{
    const std::string tmp = path + ".tmp";
    for (int attempt = 0; attempt < 6; ++attempt)
    {
        FILE *fp = OpenSharedWrite(tmp);
        if (fp == nullptr)
        {
            break;
        }
        std::fwrite(content.data(), 1, content.size(), fp);
        std::fclose(fp);

        std::error_code ec;
        std::filesystem::rename(tmp, path, ec);
        if (!ec)
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    FILE *fp = OpenSharedWrite(path);
    if (fp == nullptr)
    {
        return false;
    }
    std::fwrite(content.data(), 1, content.size(), fp);
    std::fclose(fp);
    return true;
}
} // namespace

Recorder::~Recorder()
{
    if (m_state == "running" && !m_runDir.empty())
    {
        // 没有正常收尾（异常退出路径），留下诚实的终态.
        EndRun("crashed", "recorder destroyed without EndRun");
        return;
    }
    if (m_scalars != nullptr)
    {
        std::fclose(m_scalars);
    }
    if (m_histograms != nullptr)
    {
        std::fclose(m_histograms);
    }
    if (m_activations != nullptr)
    {
        std::fclose(m_activations);
    }
}

bool Recorder::BeginRun(const std::string &runsRoot, const RunMeta &meta,
                        const LoggingPolicy &policy, const std::vector<ProbeRef> &probes,
                        const std::vector<std::vector<double>> &probePixels, uint32_t probeWidth)
{
    m_policy = policy;
    m_meta = meta;
    m_runId = MakeRunId();
    m_createdAt = IsoNow();

    std::error_code ec;
    std::filesystem::create_directories(runsRoot, ec);
    if (ec)
    {
        std::printf("Recorder: 无法创建 runs 根目录 %s: %s\n", runsRoot.c_str(), ec.message().c_str());
        return false;
    }

    const std::string base = meta.expName.empty() ? std::string("exp") : meta.expName;
    const std::string stamp = TimestampForDir();
    std::string name = base + "_" + stamp;
    std::string dir = runsRoot + "/" + name;
    for (int suffix = 2; std::filesystem::exists(dir) && suffix < 1000; ++suffix)
    {
        name = base + "_" + stamp + "_" + std::to_string(suffix);
        dir = runsRoot + "/" + name;
    }

    std::filesystem::create_directories(dir, ec);
    if (ec)
    {
        std::printf("Recorder: 无法创建 run 目录 %s: %s\n", dir.c_str(), ec.message().c_str());
        return false;
    }
    m_runName = name;
    m_runDir = dir;
    m_state = "running";

    WriteMeta(meta, probes);
    WriteProbeInputs(probePixels, probeWidth);

    m_scalars = OpenSharedWrite(dir + "/scalars.jsonl");
    m_histograms = OpenSharedWrite(dir + "/histograms.bin");
    m_activations = OpenSharedWrite(dir + "/activations.bin");
    if (m_scalars == nullptr || m_histograms == nullptr || m_activations == nullptr)
    {
        std::printf("Recorder: 无法创建数据文件于 %s\n", dir.c_str());
        return false;
    }

    const uint32_t layerCount = static_cast<uint32_t>(meta.layers.size());
    const uint32_t bins = m_policy.histBins;

    // histograms.bin 头：24 + 16 * 层数 字节
    const uint32_t histRecordBytes = 4 + layerCount * (2 + bins) * 4;
    WriteU32(m_histograms, Magic('T', 'N', 'H', '1'));
    WriteU32(m_histograms, 1);
    WriteU32(m_histograms, histRecordBytes);
    WriteU32(m_histograms, layerCount);
    WriteU32(m_histograms, bins);
    WriteU32(m_histograms, 0);
    for (size_t i = 0; i < meta.layers.size(); ++i)
    {
        const double range = m_policy.RangeFor(i);
        WriteU32(m_histograms, meta.layers[i].input);
        WriteU32(m_histograms, meta.layers[i].output);
        WriteF32(m_histograms, static_cast<float>(-range));
        WriteF32(m_histograms, static_cast<float>(range));
    }

    // activations.bin 头：16 + 4 * 层数 字节
    uint32_t actRecordBytes = 4;
    for (const auto &layer : meta.layers)
    {
        actRecordBytes += layer.output * 4;
    }
    WriteU32(m_activations, Magic('T', 'N', 'A', '1'));
    WriteU32(m_activations, 1);
    WriteU32(m_activations, actRecordBytes);
    WriteU32(m_activations, layerCount);
    for (const auto &layer : meta.layers)
    {
        WriteU32(m_activations, layer.output);
    }

    m_startMs = NowSteadyMs();
    m_lastFlushMs = m_startMs;
    m_lastHeartbeatMs = m_startMs;
    m_lastStep = 0;

    WriteStatus(0, "running", std::string());
    FlushIfDue(true);
    return true;
}

void Recorder::LogScalars(const ScalarRecord &record)
{
    if (m_scalars == nullptr)
    {
        return;
    }
    m_lastStep = record.step;

    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "{\"step\":%u,\"samplesSeen\":%llu,\"tMs\":%s,\"wallMs\":%lld,\"lr\":%s,"
                  "\"loss\":%s,\"gradNorm\":%s,\"weightNorm\":%s,\"updateRatio\":%s}\n",
                  record.step, static_cast<unsigned long long>(record.samplesSeen),
                  FmtNumber(record.elapsedMs).c_str(), static_cast<long long>(record.wallMs),
                  FmtNumber(record.lr).c_str(), FmtNumber(record.loss).c_str(),
                  FmtNumber(record.gradNorm).c_str(), FmtNumber(record.weightNorm).c_str(),
                  FmtNumber(record.updateRatio).c_str());
    std::fwrite(buf, 1, std::strlen(buf), m_scalars);
    FlushIfDue(false);
}

void Recorder::LogHistograms(uint32_t step, const std::vector<TnLayer *> &layers)
{
    if (m_histograms == nullptr || layers.empty())
    {
        return;
    }
    const uint32_t bins = m_policy.histBins;
    const size_t layerCount = layers.size();

    if (m_histCounts.size() != layerCount * bins)
    {
        m_histCounts.assign(layerCount * bins, 0.0f);
    }

    std::vector<float> mins(layerCount, 0.0f);
    std::vector<float> maxs(layerCount, 0.0f);

    for (size_t l = 0; l < layerCount; ++l)
    {
        // 每层可以有自己的固定范围：小 fan-in 的输出层权重量级和首层差很多
        const double range = m_policy.RangeFor(l);
        const double lo = -range;
        const double span = (range > 0.0) ? (2.0 * range) : 1.0;

        std::fill(m_histCounts.begin() + l * bins, m_histCounts.begin() + (l + 1) * bins, 0.0f);
        const TnLayer &layer = *layers[l];
        double mn = 0.0;
        double mx = 0.0;
        bool first = true;

        auto accumulate = [&](double v) {
            if (first)
            {
                mn = v;
                mx = v;
                first = false;
            }
            else
            {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            int idx = static_cast<int>(std::floor((v - lo) / span * static_cast<double>(bins)));
            idx = std::max(idx, 0);
            if (idx >= static_cast<int>(bins))
            {
                idx = static_cast<int>(bins) - 1;
            }
            m_histCounts[l * bins + idx] += 1.0f;
        };

        for (const auto &row : layer.Matrix())
        {
            for (double v : row)
            {
                accumulate(v);
            }
        }
        for (double v : layer.Bias())
        {
            accumulate(v);
        }

        mins[l] = std::isfinite(mn) ? static_cast<float>(mn) : 0.0f;
        maxs[l] = std::isfinite(mx) ? static_cast<float>(mx) : 0.0f;
    }

    WriteU32(m_histograms, step);
    for (size_t l = 0; l < layerCount; ++l)
    {
        WriteF32(m_histograms, mins[l]);
        WriteF32(m_histograms, maxs[l]);
        for (uint32_t b = 0; b < bins; ++b)
        {
            WriteF32(m_histograms, m_histCounts[l * bins + b]);
        }
    }
    FlushIfDue(false);
}

void Recorder::LogActivations(uint32_t step, uint32_t probeIndex,
                              const std::vector<TnLayer *> &layers)
{
    (void) probeIndex; // 记录顺序即 meta.probes 的顺序，不需要额外字段
    if (m_activations == nullptr)
    {
        return;
    }

    WriteU32(m_activations, step);
    for (size_t l = 0; l < layers.size(); ++l)
    {
        const TnVector &values = layers[l]->Values();
        const uint32_t declared =
            (l < m_meta.layers.size()) ? m_meta.layers[l].output : static_cast<uint32_t>(values.size());
        for (uint32_t i = 0; i < declared; ++i)
        {
            float v = 0.0f;
            if (i < values.size() && std::isfinite(values[i]))
            {
                v = static_cast<float>(values[i]);
            }
            WriteF32(m_activations, v);
        }
    }
    FlushIfDue(false);
}

void Recorder::Heartbeat(uint32_t lastStep)
{
    m_lastStep = lastStep;
    const uint64_t now = NowSteadyMs();
    if (now - m_lastHeartbeatMs >= m_policy.heartbeatMs)
    {
        m_lastHeartbeatMs = now;
        WriteStatus(lastStep, "running", std::string());
    }
    FlushIfDue(false);
}

void Recorder::EndRun(const char *state, const std::string &error)
{
    FlushIfDue(true);
    if (m_scalars != nullptr)
    {
        std::fclose(m_scalars);
        m_scalars = nullptr;
    }
    if (m_histograms != nullptr)
    {
        std::fclose(m_histograms);
        m_histograms = nullptr;
    }
    if (m_activations != nullptr)
    {
        std::fclose(m_activations);
        m_activations = nullptr;
    }
    if (!m_runDir.empty())
    {
        WriteStatus(m_lastStep, state, error);
        m_state = state;
    }
}

void Recorder::SetUrl(const std::string &url)
{
    m_url = url;
    if (!m_runDir.empty())
    {
        WriteStatus(m_lastStep, m_state.c_str(), std::string());
    }
}

void Recorder::SetFinalAccuracy(double accuracy)
{
    m_hasAccuracy = std::isfinite(accuracy);
    m_accuracy = m_hasAccuracy ? accuracy : 0.0;
    if (!m_runDir.empty())
    {
        WriteStatus(m_lastStep, m_state.c_str(), std::string());
    }
}

void Recorder::FlushIfDue(bool force)
{
    const uint64_t now = NowSteadyMs();
    if (!force && now - m_lastFlushMs < m_policy.flushMs)
    {
        return;
    }
    // 只需要 fflush：HTTP 服务和训练进程共享 page cache，fflush 之后读者立刻可见。
    // 掉电安全需要 FlushFileBuffers，但本场景不需要。
    if (m_scalars != nullptr)
    {
        std::fflush(m_scalars);
    }
    if (m_histograms != nullptr)
    {
        std::fflush(m_histograms);
    }
    if (m_activations != nullptr)
    {
        std::fflush(m_activations);
    }
    m_lastFlushMs = now;
}

void Recorder::WriteStatus(uint32_t lastStep, const char *state, const std::string &error)
{
    std::string j = "{";
    j += "\"runId\":\"" + m_runId + "\",";
    j += "\"state\":\"" + JsonEscape(state) + "\",";
    j += "\"lastStep\":" + std::to_string(lastStep) + ",";
    j += "\"totalSteps\":" + std::to_string(m_meta.totalSteps) + ",";
    j += "\"accuracy\":" + (m_hasAccuracy ? FmtNumber(m_accuracy) : std::string("null")) + ",";
    j += "\"heartbeatMs\":" + std::to_string(NowWallMs()) + ",";
    j += "\"url\":\"" + JsonEscape(m_url) + "\",";
    j += "\"error\":";
    j += error.empty() ? std::string("null") : ("\"" + JsonEscape(error) + "\"");
    j += "}\n";
    WriteFileAtomic(m_runDir + "/status.json", j);
}

void Recorder::WriteProbeInputs(const std::vector<std::vector<double>> &probePixels,
                                uint32_t probeWidth)
{
    if (m_runDir.empty() || probePixels.empty() || probeWidth == 0)
    {
        return;
    }
    std::string path = m_runDir + "/probe_inputs.bin";
    FILE *fp = OpenSharedWrite(path);
    if (fp == nullptr)
    {
        return;
    }
    const uint32_t count = static_cast<uint32_t>(probePixels.size());
    WriteU32(fp, Magic('T', 'N', 'P', '1'));
    WriteU32(fp, 1);
    WriteU32(fp, count);
    WriteU32(fp, probeWidth);
    for (const auto &pixels : probePixels)
    {
        for (uint32_t i = 0; i < probeWidth; ++i)
        {
            double v = (i < pixels.size()) ? pixels[i] : 0.0;
            WriteF32(fp, std::isfinite(v) ? static_cast<float>(v) : 0.0f);
        }
    }
    std::fclose(fp);
}

void Recorder::WriteMeta(const RunMeta &meta, const std::vector<ProbeRef> &probes)
{
    std::string j;
    j += "{\n";
    j += "  \"formatVersion\": 1,\n";
    j += "  \"runId\": \"" + m_runId + "\",\n";
    j += "  \"createdAt\": \"" + m_createdAt + "\",\n";
    j += "  \"expName\": \"" + JsonEscape(meta.expName) + "\",\n";
    j += "  \"seed\": " + std::to_string(meta.seed) + ",\n";

    j += "  \"dataset\": {\"path\": \"" + JsonEscape(meta.datasetPath) +
         "\", \"trainCount\": " + std::to_string(meta.trainCount) +
         ", \"testCount\": " + std::to_string(meta.testCount) + "},\n";

    j += "  \"network\": {\n";
    j += "    \"cost\": \"" + JsonEscape(meta.costFunction) + "\",\n";
    j += "    \"layers\": [";
    for (size_t i = 0; i < meta.layers.size(); ++i)
    {
        const LayerDesc &layer = meta.layers[i];
        j += "{\"in\": " + std::to_string(layer.input) + ", \"out\": " +
             std::to_string(layer.output) + ", \"activation\": \"" + JsonEscape(layer.activation) +
             "\"}";
        if (i + 1 < meta.layers.size())
        {
            j += ", ";
        }
    }
    j += "]\n  },\n";

    j += "  \"hyperparams\": {\"batchSize\": " + std::to_string(meta.batchSize) +
         ", \"totalSteps\": " + std::to_string(meta.totalSteps) + ", \"lrSchedule\": [";
    j += "{\"fromStep\": 1, \"value\": " + FmtNumber(meta.lrHigh) + "}";
    if (meta.lrDecayFromStep > 0 && meta.lrDecayFromStep <= meta.totalSteps)
    {
        j += ", {\"fromStep\": " + std::to_string(meta.lrDecayFromStep) +
             ", \"value\": " + FmtNumber(meta.lrLow) + "}";
    }
    j += "]},\n";

    j += "  \"logging\": {\"scalarsEvery\": " + std::to_string(m_policy.scalarsEvery) +
         ", \"histEvery\": " + std::to_string(m_policy.histEvery) +
         ", \"actEvery\": " + std::to_string(m_policy.actEvery) +
         ", \"histBins\": " + std::to_string(m_policy.histBins) + ", \"histRange\": [";
    for (size_t i = 0; i < meta.layers.size(); ++i)
    {
        const double range = m_policy.RangeFor(i);
        j += "[" + FmtNumber(-range) + ", " + FmtNumber(range) + "]";
        if (i + 1 < meta.layers.size())
        {
            j += ", ";
        }
    }
    j += "]},\n";

    j += "  \"probes\": [";
    for (size_t i = 0; i < probes.size(); ++i)
    {
        j += "{\"index\": " + std::to_string(probes[i].index) +
             ", \"label\": " + std::to_string(probes[i].label) + ", \"source\": \"" +
             JsonEscape(probes[i].source) + "\"}";
        if (i + 1 < probes.size())
        {
            j += ", ";
        }
    }
    j += "],\n";

    // 布局信息只是给前端做交叉校验用的，真正的权威来源仍是各文件自己的头.
    const uint32_t layerCount = static_cast<uint32_t>(meta.layers.size());
    const uint32_t bins = m_policy.histBins;
    j += "  \"histogramLayout\": {\"headerBytes\": " + std::to_string(24 + 16 * layerCount) +
         ", \"recordBytes\": " + std::to_string(4 + layerCount * (2 + bins) * 4) + "},\n";

    uint32_t actRecordBytes = 4;
    for (const auto &layer : meta.layers)
    {
        actRecordBytes += layer.output * 4;
    }
    j += "  \"activationLayout\": {\"headerBytes\": " + std::to_string(16 + 4 * layerCount) +
         ", \"recordBytes\": " + std::to_string(actRecordBytes) + "},\n";

    j += "  \"scalarFields\": [";
    j += "{\"key\": \"loss\", \"label\": \"Loss\", \"logScale\": true}";
    j += ", {\"key\": \"lr\", \"label\": \"Learning rate\", \"logScale\": false}";
    j += ", {\"key\": \"gradNorm\", \"label\": \"Grad norm\", \"logScale\": true}";
    j += ", {\"key\": \"weightNorm\", \"label\": \"Weight norm\", \"logScale\": false}";
    j += ", {\"key\": \"updateRatio\", \"label\": \"Update ratio\", \"logScale\": true}";
    j += "]\n";
    j += "}\n";

    WriteFileAtomic(m_runDir + "/meta.json", j);
}
