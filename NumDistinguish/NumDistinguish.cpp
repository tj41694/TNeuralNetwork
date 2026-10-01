#include "NumDistinguish.h"
#include "Sample.h"
#include "Shuffle.h"
#include <cassert>

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

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, ActiveFuncPtr activeFunc,
                                   DerivFuncPtr derivFunc)
{
    TnLayer *layer = new TnLayer(row, colum, activeFunc, derivFunc);
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
        UpdateWeights(gradients, batchs.size(), lRate);
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
    int corectCount = 0;
    for (auto s : data)
    {
        int num = Distinguish(*s);
        if (num == s->m_realValue)
            corectCount++;
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
        // 当前层的预激活值 z_i，用于计算激活函数的导数
        const TnVector &preActiveVals = m_layers[i]->PreActiveValues();
        TnVector *preGradients = i == 0 ? nullptr : &gradientLayers[i - 1]->Values();
        gradientLayers[i]->CalcGradient(*prevLyActiveVals, preActiveVals, m_layers[i]->matrix,
                                        preGradients);
    }
}

void DigitalDistinguish::UpdateWeights(const vector<TnLayer *> &gradients, size_t batchSize,
                                       double stepRate)
{
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        for (size_t r = 0; r < gradients[i]->matrix.size(); r++)
        {
            m_layers[i]->bias[r] -= (stepRate * gradients[i]->bias[r] / batchSize);
            for (size_t c = 0; c < gradients[i]->matrix[r].size(); c++)
            {
                m_layers[i]->matrix[r][c] -= (stepRate * gradients[i]->matrix[r][c] / batchSize);
            }
        }
    }
}

DigitalDistinguish::~DigitalDistinguish()
{
    for (auto layer : m_layers)
    {
        delete layer;
    }
}
