#include "NumDistinguish.h"
#include "Sample.h"
#include "Shuffle.h"
#include <cassert>
#include <filesystem>
#include <string>

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
        if (val < 0)
            val = 0;
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

void DigitalDistinguish::PushLayer(unsigned int input, unsigned int output,
                                   ActiveFuncPtr activeFunc, DerivFuncPtr derivFunc)
{
    TnLayer *layer = new TnLayer(output, input, activeFunc, derivFunc);
    m_layers.emplace_back(layer);
}

void DigitalDistinguish::Training(const vector<Sample *> &samples, int batchSize)
{
    Shuffle shuff(samples.size());
    int count = batchSize;
    int times = 0;
    double lRate = 0.1;
    while (times++ < 25000)
    {
        vector<Sample *> batchs;
        shuff.GetShuffledData(samples, batchSize, batchs);
        double sampleTotalVal = 0;
        vector<TnLayer *> gradients;
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
        UpdateWeights(gradients, batchs.size(),  times < 20000 ? lRate : 0.05);
        for (auto gradient : gradients)
            delete gradient;
        printf("Sample Count: %d \t lRate: %.10f\tCost Value: %.5f \n", count, lRate,
               sampleTotalVal / batchSize);
        count += batchSize;
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
    for (auto s : data)
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
    double corectRate = 100.0 * (double) corectCount / data.size();
    printf("accuracy: %.3f%%\n", corectRate);
}

// 每一组 batch 多次 backward 共用相同的 gradient layers，但注意每一次 backward 都会重写 gradient
// layer 的 value。基于此梯度值对共用 gradient layer 的权重以及偏置进行累加。
// 即 gradient layers 内的权重以及偏置是累加共用的，但其内部的 value 值是每一次 backward 重新写入的
void DigitalDistinguish::Backward(const Sample &input, const TnVector &output,
                                  vector<TnLayer *> &gradientLayers) const
{
    assert(gradientLayers.size() == m_layers.size());
    int layerCt = m_layers.size();

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
    double ratio = stepRate / batchSize;
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        *gradients[i] *= ratio;
        *m_layers[i] -= *gradients[i];
    }
}

DigitalDistinguish::~DigitalDistinguish()
{
    for (auto layer : m_layers)
    {
        delete layer;
    }
}
