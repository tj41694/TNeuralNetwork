#include "NumDistinguish.h"
#include "Sample.h"
#include "Shuffle.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace
{
uint64_t SteadyMs()
{
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count());
}

int64_t WallMs()
{
    using namespace std::chrono;
    return static_cast<int64_t>(
        duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

// 一组层的整体 L2 范数：sqrt(Σ ‖layer‖²)，梯度与权重的范数统计共用.
double TotalNorm(const vector<TnLayer *> &layers)
{
    double sum = 0;
    for (const auto *layer : layers)
    {
        sum += layer->NormSquared();
    }
    return sqrt(sum);
}

// 在给定的一串层上跑一次前向，不依赖 DigitalDistinguish 的成员状态.
void ForwardLayers(const vector<TnLayer *> &layers, const TnVector &input)
{
    const TnVector *in = &input;
    for (auto *layer : layers)
    {
        *layer *= *in;
        in = &layer->Values();
    }
}

// 统计一批样本的分类准确率（0~100）.
double EvaluateAccuracy(const vector<TnLayer *> &layers, const vector<Sample *> &data)
{
    if (layers.empty() || data.empty())
    {
        return 0.0;
    }
    int correct = 0;
    for (const auto *sample : data)
    {
        ForwardLayers(layers, *sample);
        const TnVector &out = layers.back()->Values();
        int best = 0;
        for (int i = 1; i < (int) out.size(); ++i)
        {
            if (out[i] > out[best])
            {
                best = i;
            }
        }
        if (best == sample->m_realValue)
        {
            ++correct;
        }
    }
    return 100.0 * static_cast<double>(correct) / static_cast<double>(data.size());
}

// 周期性评估的工作线程：训练线程提交权重快照，本线程在独立网络上评估训练集与验证集，
// 结果放回由训练线程写遥测。评估只读自己的快照，绝不触碰训练中的 m_layers，因此无需锁网络.
class ValidationWorker
{
  public:
    ValidationWorker(const vector<Sample *> &trainData, const vector<Sample *> &testData)
        : m_train(trainData), m_test(testData)
    {
    }

    ~ValidationWorker()
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_thread.joinable())
        {
            m_thread.join();
        }
        Clear(m_pending);
    }

    void Start()
    {
        m_thread = std::thread([this]() { Run(); });
    }

    // 训练线程调用：对当前权重做快照。若上一份还没评估完，丢弃旧的只保留最新，避免堆积.
    void Submit(uint32_t step, const vector<TnLayer *> &layers)
    {
        vector<TnLayer *> snapshot;
        snapshot.reserve(layers.size());
        for (const auto *layer : layers)
        {
            snapshot.push_back(new TnLayer(layer->Clone()));
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            Clear(m_pending);
            m_pending = std::move(snapshot);
            m_pendingStep = step;
            m_hasPending = true;
        }
        m_cv.notify_one();
    }

    // 训练线程调用：取走一个已完成的结果（非阻塞）.
    bool TryPop(uint32_t &step, double &trainAcc, double &testAcc)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_hasResult)
        {
            return false;
        }
        step = m_resultStep;
        trainAcc = m_trainAcc;
        testAcc = m_testAcc;
        m_hasResult = false;
        return true;
    }

    // 训练结束时调用：让线程把手上这份快照评估完再退出，并返回最后一个结果.
    bool StopAndDrain(uint32_t &step, double &trainAcc, double &testAcc)
    {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        if (m_thread.joinable())
        {
            m_thread.join();
        }
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_hasResult)
        {
            return false;
        }
        step = m_resultStep;
        trainAcc = m_trainAcc;
        testAcc = m_testAcc;
        m_hasResult = false;
        return true;
    }

  private:
    static void Clear(vector<TnLayer *> &layers)
    {
        for (auto *layer : layers)
        {
            delete layer;
        }
        layers.clear();
    }

    void Run()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        while (true)
        {
            m_cv.wait(lock, [this]() { return m_stop || m_hasPending; });
            if (!m_hasPending)
            {
                // 停止且没有待评估的快照，退出
                break;
            }
            vector<TnLayer *> snapshot = std::move(m_pending);
            const uint32_t step = m_pendingStep;
            m_hasPending = false;
            lock.unlock();

            // 在一个不被打扰的局部网络上评估
            const double trainAcc = EvaluateAccuracy(snapshot, m_train);
            const double testAcc = EvaluateAccuracy(snapshot, m_test);
            Clear(snapshot);

            lock.lock();
            m_resultStep = step;
            m_trainAcc = trainAcc;
            m_testAcc = testAcc;
            m_hasResult = true;
        }
    }

    const vector<Sample *> &m_train;
    const vector<Sample *> &m_test;
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    bool m_stop = false;
    bool m_hasPending = false;
    bool m_hasResult = false;
    uint32_t m_pendingStep = 0;
    uint32_t m_resultStep = 0;
    double m_trainAcc = 0.0;
    double m_testAcc = 0.0;
    vector<TnLayer *> m_pending;
};
} // namespace

void Sigmoid(TnVector &input)
{
    // TODO
}
void DerivSigmoid(const TnVector &preActiveValues, TnVector &vec)
{
    // TODO
}

void ReLU(TnVector &input)
{
    for (auto &val : input)
    {
        val = std::max<double>(val, 0);
    }
}
void DerivReLU(const TnVector &preActiveValues, TnVector &vec)
{
    for (size_t i = 0; i < vec.size(); i++)
    {
        if (preActiveValues[i] <= 0)
            vec[i] = 0;
    }
}

void SoftMax(TnVector &input)
{
    if (input.empty())
        return;
    // 先减去最大值再取指数，避免预激活过大时 exp 溢出成 inf/inf 得到 NaN.
    double maxVal = input[0];
    for (double val : input)
    {
        if (val > maxVal)
            maxVal = val;
    }
    double total = 0;
    for (auto &val : input)
    {
        val = exp(val - maxVal);
        total += val;
    }
    for (auto &val : input) // 归一化
    {
        val /= total;
    }
}
void DerivSoftMax(const TnVector &preActiveValues, TnVector &vec)
{
}

void DigitalDistinguish::PushLayer(int input, int output,
                                   ActiveFuncPtr activeFunc, DerivFuncPtr derivFunc)
{
    TnLayer *layer = new TnLayer(output, input, activeFunc, derivFunc);
    m_layers.emplace_back(layer);
}

void DigitalDistinguish::Training(const vector<Sample *> &samples, const TrainingOptions &options)
{
    Shuffle shuff(samples.size());
    const int batchSize = options.batchSize > 0 ? options.batchSize : 1;
    Recorder *recorder = options.recorder;
    const LoggingPolicy *policy = (recorder != nullptr) ? &recorder->Policy() : nullptr;
    const uint32_t printEvery = (policy != nullptr) ? policy->printEvery : 0;

    const uint64_t startMs = SteadyMs();
    uint64_t samplesSeen = 0;

    for (uint32_t step = 1; step <= options.totalSteps; ++step)
    {
        if (options.stopRequested != nullptr && options.stopRequested->load())
        {
            break;
        }

        vector<Sample *> batchs;
        shuff.GetShuffledData(samples, batchSize, batchs);
        double sampleTotalVal = 0;
        vector<TnLayer *> gradients;
        gradients.reserve(m_layers.size());
        for (const auto &layer : m_layers)
            gradients.emplace_back(new TnLayer(*layer));
        for (Sample *batch : batchs)
        {
            const auto &input = *batch;
            Forward(input);
            const auto &output = m_layers.back()->Values();
            sampleTotalVal += input.GetCostValue(CostFunc::CrossEntropy, output);
            Backward(input, output, gradients, batchs.size());
        }

        // 此时梯度层是 batch 平均后的梯度；先取范数再更新权重.
        double gradNorm = TotalNorm(gradients);

        const double lr = (step >= options.lrDecayFromStep) ? options.lrLow : options.lrHigh;
        UpdateWeights(gradients, lr);
        for (auto *gradient : gradients)
            delete gradient;

        double weightNorm = TotalNorm(m_layers);

        const size_t effectiveBatch = batchs.empty() ? static_cast<size_t>(batchSize) : batchs.size();
        samplesSeen += effectiveBatch;
        const double loss = sampleTotalVal / static_cast<double>(effectiveBatch);

        if (recorder != nullptr)
        {
            ScalarRecord record;
            record.step = step;
            record.samplesSeen = samplesSeen;
            record.elapsedMs = static_cast<double>(SteadyMs() - startMs);
            record.wallMs = WallMs();
            record.lr = lr;
            record.loss = loss;
            record.gradNorm = gradNorm;
            record.weightNorm = weightNorm;
            record.updateRatio =
                (weightNorm > 0.0) ? lr * gradNorm / weightNorm : 0.0;
            recorder->LogScalars(record);

            if (policy->histEvery > 0 && step % policy->histEvery == 0)
            {
                recorder->LogHistograms(step, m_layers);
            }

            if (policy->actEvery > 0 && step % policy->actEvery == 0 &&
                options.probes != nullptr && !options.probes->empty())
            {
                // 权重与激活必须对应同一个 step，所以在 UpdateWeights 之后单独跑一次
                // 前向。它会覆盖 m_layers 的 values，但下一轮会完整重算，无害.
                for (size_t i = 0; i < options.probes->size(); ++i)
                {
                    Forward(*(*options.probes)[i]);
                    recorder->LogActivations(step, static_cast<uint32_t>(i), m_layers);
                }
            }
            recorder->Heartbeat(step);
        }

        if (printEvery == 0 || step % printEvery == 0 || step == options.totalSteps)
        {
            printf("Step: %u \t Samples: %llu \t lRate: %.6f \t Cost Value: %.5f\n", step,
                   static_cast<unsigned long long>(samplesSeen), lr, loss);
        }
    }
}

void DigitalDistinguish::InitAdamState()
{
    // 成员 m_adamGradients 保存跨 step 的 Adam 一阶动量，拷贝构造会零初始化，正合适.
    for (auto *gradient : m_adamGradients)
        delete gradient;
    m_adamGradients.clear();
    m_adamGradients.reserve(m_layers.size());
    for (const auto &layer : m_layers)
        m_adamGradients.emplace_back(new TnLayer(*layer));

    // 逐参数学习率同样按 m_layers 的结构初始化，拷贝构造零初始化.
    for (auto *rate : m_adamLearningRates)
        delete rate;
    m_adamLearningRates.clear();
    m_adamLearningRates.reserve(m_layers.size());
    for (const auto &layer : m_layers)
        m_adamLearningRates.emplace_back(new TnLayer(*layer));
}

void DigitalDistinguish::TrainingAdam(const vector<Sample *> &samples,
                                      const TrainingOptions &options, double momentumBeta,
                                      double rsmBeta)
{
    InitAdamState();

    Shuffle shuff(samples.size());
    const int batchSize = options.batchSize > 0 ? options.batchSize : 1;
    Recorder *recorder = options.recorder;
    const LoggingPolicy *policy = (recorder != nullptr) ? &recorder->Policy() : nullptr;
    const uint32_t printEvery = (policy != nullptr) ? policy->printEvery : 0;

    // 周期性准确率评估跑在独立线程上：训练线程只做权重快照与写遥测，
    // 评估在本线程之外完成，不会读写训练中的 m_layers.
    std::unique_ptr<ValidationWorker> validator;
    if (recorder != nullptr && options.validationData != nullptr &&
        !options.validationData->empty() && options.metricsEvery > 0)
    {
        validator = std::make_unique<ValidationWorker>(samples, *options.validationData);
        validator->Start();
    }

    const uint64_t startMs = SteadyMs();
    uint64_t samplesSeen = 0;

    for (uint32_t step = 1; step <= options.totalSteps; ++step)
    {
        if (options.stopRequested != nullptr && options.stopRequested->load())
        {
            break;
        }

        vector<Sample *> batchs;
        shuff.GetShuffledData(samples, batchSize, batchs);
        double sampleTotalVal = 0;
        vector<TnLayer *> gradients;
        gradients.reserve(m_layers.size());
        for (const auto &layer : m_layers)
            gradients.emplace_back(new TnLayer(*layer));
        for (Sample *batch : batchs)
        {
            const auto &input = *batch;
            Forward(input);
            const auto &output = m_layers.back()->Values();
            sampleTotalVal += input.GetCostValue(CostFunc::CrossEntropy, output);
            Backward(input, output, gradients, batchs.size());
        }

        // 一个 batch 反向传播完毕：把成员里保存的动量与当前 batch 计算出的梯度融合.
        FuseGradients(gradients, momentumBeta, rsmBeta);
        for (auto *gradient : gradients)
            delete gradient;

        // 融合后的动量即为本步用于更新的梯度，取范数后再更新权重.
        double gradNorm = TotalNorm(m_adamGradients);

        const double lr = options.lrHigh / 100;
        UpdateWeightsAdam(lr, step, momentumBeta, rsmBeta);

        double weightNorm = TotalNorm(m_layers);

        const size_t effectiveBatch = batchs.empty() ? static_cast<size_t>(batchSize) : batchs.size();
        samplesSeen += effectiveBatch;
        const double loss = sampleTotalVal / static_cast<double>(effectiveBatch);

        if (recorder != nullptr)
        {
            ScalarRecord record;
            record.step = step;
            record.samplesSeen = samplesSeen;
            record.elapsedMs = static_cast<double>(SteadyMs() - startMs);
            record.wallMs = WallMs();
            record.lr = lr;
            record.loss = loss;
            record.gradNorm = gradNorm;
            record.weightNorm = weightNorm;
            record.updateRatio =
                (weightNorm > 0.0) ? lr * gradNorm / weightNorm : 0.0;
            recorder->LogScalars(record);

            if (policy->histEvery > 0 && step % policy->histEvery == 0)
            {
                recorder->LogHistograms(step, m_layers);
            }

            if (policy->actEvery > 0 && step % policy->actEvery == 0 &&
                options.probes != nullptr && !options.probes->empty())
            {
                // 权重与激活必须对应同一个 step，所以在 UpdateWeights 之后单独跑一次
                // 前向。它会覆盖 m_layers 的 values，但下一轮会完整重算，无害.
                for (size_t i = 0; i < options.probes->size(); ++i)
                {
                    Forward(*(*options.probes)[i]);
                    recorder->LogActivations(step, static_cast<uint32_t>(i), m_layers);
                }
            }
            if (validator)
            {
                if (step % options.metricsEvery == 0)
                {
                    validator->Submit(step, m_layers);
                }
                uint32_t metricStep = 0;
                double trainAcc = 0.0;
                double testAcc = 0.0;
                if (validator->TryPop(metricStep, trainAcc, testAcc))
                {
                    MetricRecord metrics;
                    metrics.step = metricStep;
                    metrics.trainAcc = trainAcc;
                    metrics.testAcc = testAcc;
                    recorder->LogMetrics(metrics);
                }
            }
            recorder->Heartbeat(step);
        }

        if (printEvery == 0 || step % printEvery == 0 || step == options.totalSteps)
        {
            printf("Step: %u \t Samples: %llu \t lRate: %.6f \t Cost Value: %.5f\n", step,
                   static_cast<unsigned long long>(samplesSeen), lr, loss);
        }
    }

    // 收尾：让评估线程把手上的快照算完，补写最后一条准确率，避免短 run 一条都没有.
    if (validator != nullptr && recorder != nullptr)
    {
        uint32_t metricStep = 0;
        double trainAcc = 0.0;
        double testAcc = 0.0;
        if (validator->StopAndDrain(metricStep, trainAcc, testAcc))
        {
            MetricRecord metrics;
            metrics.step = metricStep;
            metrics.trainAcc = trainAcc;
            metrics.testAcc = testAcc;
            recorder->LogMetrics(metrics);
        }
    }
}

// 融合 m_adamGradients / m_adamLearningRates（成员里保存的一阶动量与二阶矩）与
// currentGradients（当前 batch 计算的梯度）：m = β1·m + (1−β1)·g，v = β2·v + (1−β2)·g².
// 偏差修正不在这里做，否则修正后的值下一步会被再修正一次，逐级放大导致发散.
void DigitalDistinguish::FuseGradients(const vector<TnLayer *> &currentGradients,
                                       double momentumRatio, double rsmRatio)
{
    assert(m_adamGradients.size() == currentGradients.size());
    assert(m_adamLearningRates.size() == currentGradients.size());

    const double oneMinusBeta = 1.0 - momentumRatio;
    for (size_t l = 0; l < m_adamGradients.size(); ++l)
    {
        *m_adamGradients[l] *= momentumRatio;
        m_adamGradients[l]->AddScaled(*currentGradients[l], oneMinusBeta);
    }

    // Adam 二阶矩：v = β2·v + (1−β2)·g²，逐参数保存平方梯度的指数滑动平均.
    const double oneMinusRsm = 1.0 - rsmRatio;
    for (size_t l = 0; l < m_adamLearningRates.size(); ++l)
    {
        *m_adamLearningRates[l] *= rsmRatio;
        m_adamLearningRates[l]->AddScaled(currentGradients[l]->Square(), oneMinusRsm);
    }
}

void DigitalDistinguish::Forward(const Sample &input)
{
    for (int i = 0; i < (int) m_layers.size(); ++i)
    {
        auto &layer = *m_layers[i];
        if (i == 0)
            layer *= input;
        else
        {
            auto &preLayer = *m_layers[i - 1];
            layer *= preLayer.Values();
        }
    }
}

int DigitalDistinguish::Distinguish(const Sample &sample)
{
    Forward(sample);
    double v = -1;
    int result = -1;
    for (int i = 0; i < (int) m_layers.back()->Values().size(); i++)
    {
        if (v < m_layers.back()->Values()[i])
        {
            result = i;
            v = m_layers.back()->Values()[i];
        }
    }
    return result;
}

double DigitalDistinguish::Validate(const vector<Sample *> &data)
{
    // const char *errorDir = "errors";
    // std::filesystem::create_directories(errorDir);
    int corectCount = 0;
    int index = 0;
    for (auto *s : data)
    {
        int num = Distinguish(*s);
        if (num == s->m_realValue)
            corectCount++;
        else
        {
            // string path = string(errorDir) + "/" + to_string(index) + "_pred" + to_string(num) +
            //               "_real" + to_string(s->m_realValue) + ".bmp";
            // s->SaveAsBmp(path.c_str());
        }
        index++;
    }
    const double corectRate =
        data.empty() ? 0.0 : 100.0 * (double) corectCount / (double) data.size();
    printf("accuracy: %.3f%%\n", corectRate);
    return corectRate;
}

// 每一组 batch 多次 backward 共用相同的 gradient layers，但注意每一次 backward 都会重写 gradient
// layer 的 value。基于此梯度值对共用 gradient layer 的权重以及偏置进行累加。
// 即 gradient layers 内的权重以及偏置是累加共用的，但其内部的 value 值是每一次 backward 重新写入的
void DigitalDistinguish::Backward(const Sample &input, const TnVector &output,
                                  vector<TnLayer *> &gradientLayers, size_t batchSize) const
{
    assert(gradientLayers.size() == m_layers.size());
    int layerCt = (int)m_layers.size();

    gradientLayers.back()->Values() = output;

    // One-hot梯度值等于其自身减1, 除了one hot，其他梯度值都是激活值本身.前提是最后的输出层是用
    // softmax 加交叉熵的组合
    gradientLayers.back()->Values()[input.m_realValue] -= 1;

    // 第一时间除以 batchSize：后续整条反向链对输出层梯度都是线性缩放，逐样本累加完即为
    // batch 平均梯度，所以 UpdateWeights 里无需再除.
    const double invBatch = 1.0 / static_cast<double>(batchSize > 0 ? batchSize : 1);
    for (auto &v : gradientLayers.back()->Values())
    {
        v *= invBatch;
    }

    // 由输出层至输入层逐层反向传播
    for (int i = layerCt - 1; i >= 0; --i)
    {
        // 上一层的激活值 a_{i-1}，用于计算权重梯度
        const TnVector *prevLyActiveVals = i == 0 ? &input : &m_layers[i - 1]->Values();
        TnVector *preGradients = i == 0 ? nullptr : &gradientLayers[i - 1]->Values();
        gradientLayers[i]->CalcGradient(*prevLyActiveVals, *m_layers[i], preGradients);
    }
}

void DigitalDistinguish::UpdateWeights(const vector<TnLayer *> &gradients, double stepRate)
{
    // 梯度在反向传播时已除以 batchSize，这里只按学习率缩放后相减。
    // 用 AddScaled(负缩放) 而不是就地缩放梯度层：Adam 的成员动量在更新后必须保持原值.
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        m_layers[i]->AddScaled(*gradients[i], -stepRate);
    }
}

void DigitalDistinguish::UpdateWeightsAdam(double inputLr, uint32_t step, double momentumBeta,
                                           double rsmBeta)
{
    // Adam 偏差修正：m̂ = m/(1−β1^t)、v̂ = v/(1−β2^t)，等价于把一阶动量按 1/(1−β1^t) 放大、
    // 二阶矩按 1/(1−β2^t) 缩放到修正后的值.
    const double mCorrection = 1.0 - std::pow(momentumBeta, static_cast<double>(step));
    const double vCorrection = 1.0 - std::pow(rsmBeta, static_cast<double>(step));
    const double lr = (mCorrection > 0.0) ? inputLr / mCorrection : inputLr;
    const double vScale = (vCorrection > 0.0) ? 1.0 / vCorrection : 1.0;
    const double eps = 1e-8;

    // 逐参数自适应更新：w -= lr · m̂ / (sqrt(v̂) + eps).
    for (size_t i = 0; i < m_layers.size(); ++i)
    {
        TnLayer vHat = *m_adamLearningRates[i] * vScale;
        m_layers[i]->AddScaledNormalized(*m_adamGradients[i], vHat, -lr, eps);
    }
}

DigitalDistinguish::~DigitalDistinguish()
{
    for (auto *layer : m_layers)
    {
        delete layer;
    }
    for (auto *gradient : m_adamGradients)
    {
        delete gradient;
    }
    for (auto *rate : m_adamLearningRates)
    {
        delete rate;
    }
}
