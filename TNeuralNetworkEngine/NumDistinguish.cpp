#include "NumDistinguish.h"
#include "Sample.h"
#include "Shuffle.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

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
    double total = 0;
    for (auto &val : input)
    {
        val = exp(val);
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
            Backward(input, output, gradients);
        }

        // 必须在 UpdateWeights 之前取：那时梯度层还是"累加后的原始梯度"，
        // 一旦 UpdateWeights 跑过就被按 lr/batchSize 缩放并相减了.
        double gradNorm = TotalNorm(gradients);

        const double lr = (step >= options.lrDecayFromStep) ? options.lrLow : options.lrHigh;
        UpdateWeights(gradients, batchs.size(), lr);
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
            record.updateRatio = (weightNorm > 0.0)
                                     ? (lr / static_cast<double>(effectiveBatch)) * gradNorm /
                                           weightNorm
                                     : 0.0;
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

void DigitalDistinguish::Validate(const vector<Sample *> &data)
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
    double corectRate = 100.0 * (double) corectCount / (double)data.size();
    printf("accuracy: %.3f%%\n", corectRate);
}

// 每一组 batch 多次 backward 共用相同的 gradient layers，但注意每一次 backward 都会重写 gradient
// layer 的 value。基于此梯度值对共用 gradient layer 的权重以及偏置进行累加。
// 即 gradient layers 内的权重以及偏置是累加共用的，但其内部的 value 值是每一次 backward 重新写入的
void DigitalDistinguish::Backward(const Sample &input, const TnVector &output,
                                  vector<TnLayer *> &gradientLayers) const
{
    assert(gradientLayers.size() == m_layers.size());
    int layerCt = (int)m_layers.size();

    gradientLayers.back()->Values() = output;

    // One-hot梯度值等于其自身减1, 除了one hot，其他梯度值都是激活值本身.前提是最后的输出层是用
    // softmax 加交叉熵的组合
    gradientLayers.back()->Values()[input.m_realValue] -= 1;

    // 由输出层至输入层逐层反向传播
    for (int i = layerCt - 1; i >= 0; --i)
    {
        // 上一层的激活值 a_{i-1}，用于计算权重梯度
        const TnVector *prevLyActiveVals = i == 0 ? &input : &m_layers[i - 1]->Values();
        TnVector *preGradients = i == 0 ? nullptr : &gradientLayers[i - 1]->Values();
        gradientLayers[i]->CalcGradient(*prevLyActiveVals, *m_layers[i], preGradients);
    }
}

void DigitalDistinguish::UpdateWeights(const vector<TnLayer *> &gradients, size_t batchSize,
                                       double stepRate)
{
    double ratio = stepRate / (double)batchSize;
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        *gradients[i] *= ratio;
        *m_layers[i] -= *gradients[i];
    }
}

DigitalDistinguish::~DigitalDistinguish()
{
    for (auto *layer : m_layers)
    {
        delete layer;
    }
}
