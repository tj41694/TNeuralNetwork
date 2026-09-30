#include "NumDistinguish.h"
#include "Sample.h"
#include "Shuffle.h"

void Sigmoid(TnVector &vec)
{
    // TODO
}

void ReLU(TnVector &vec)
{
    for (auto &val : vec)
    {
        if (val < 0)
            val = 0;
    }
}

void SoftMax(TnVector &vec)
{
    double total = 0;
    for (auto &val : vec)
    {
        val = exp(val);
        total += val;
    }
    for (auto &val : vec) // 归一化
    { 
        val /= total;
    }
}

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, ActiveFuncPtr activeFunc)
{
    TnLayer *layer = new TnLayer(row, colum, activeFunc);
    m_layers.emplace_back(layer);
}

void DigitalDistinguish::Training(const vector<Sample *> &samples, int batchSize)
{
    Shuffle shuff(samples.size());
    int count = batchSize;
    int times = 0;
    while (times++ < 5000)
    {
        vector<Sample *> batchs;
        shuff.GetShuffledData(samples, batchSize, batchs);
        double sampleTotalVal = 0;
        for (Sample *sample : batchs)
        {
            ForwardPass(*sample); //TODO delete
            Sample output = *sample;
            for (const auto &layer : m_layers)
            {
                output = output * (*layer);
                layer->Active(output);
            }
            sampleTotalVal += output.GetCostValue(CostFunc::CrossEntropy);
        }
        vector<TnLayer *> gradients;
        for (const auto &layer : m_layers)
        {
            gradients.emplace_back(new TnLayer(*layer));
        }
        for (Sample *sample : batchs)
        {
            Backward(*sample, gradients);
        }
        UpdateWeights(gradients, batchs.size(), 0.001f);
        for (auto gradient : gradients)
        {
            delete gradient;
        }
        printf("Sample Count: %d \t Cost Value: %.5f \n", count, sampleTotalVal / batchSize);
        count += batchSize;
    }
}

TnVector DigitalDistinguish::ForwardPass(Sample &sample)
{
    TnVector result = sample;
    // for(const auto & layer : layers)
    // {
    //     result = result * *layer;
    //     layer->Active(result);
    // }
    size_t layerCount = m_layers.size() - 1;
    for (size_t i = 0; i < layerCount; i++)
    {
        sample.MatrixMultiply(*m_layers[i], ActiveFunc::ReLU);
    }
    sample.MatrixMultiply(*m_layers[layerCount], ActiveFunc::SoftMax);
    return result;
}

void DigitalDistinguish::InverseTrans(Sample &sample) const
{
    vector<SampleLayer> &acLayers = sample.m_activeLayers;
    acLayers.back().a[sample.m_realValue] -= 1; // 变换梯度
    for (int i = acLayers.size() - 2; i > -1; i--)
    {
        for (size_t r = 0; r < acLayers[i].a.size(); r++)
        {
            acLayers[i].a[r] = 0;
            const TnLayer &weightLayer = *m_layers[i + 1];
            for (int wr = 0; wr < weightLayer.row(); wr++)
            {
                acLayers[i].a[r] += weightLayer.matrix[wr][r] * acLayers[i + 1].a[wr];
            }
        }
    }
}

int DigitalDistinguish::Distinguish(Sample &sample)
{
    ForwardPass(sample);
    double v = -1;
    int result = -1;
    for (size_t i = 0; i < sample.m_activeLayers[sample.m_activeLayers.size() - 1].a.size(); i++)
    {
        if (v < sample.m_activeLayers[sample.m_activeLayers.size() - 1].a[i])
        {
            result = i;
            v = sample.m_activeLayers[sample.m_activeLayers.size() - 1].a[i];
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
        {
            corectCount++;
        }
        s->m_activeLayers.clear();
    }
    double corectRate = 100.0 * (double) corectCount / data.size();
    printf("accuracy: %.3f%%\n", corectRate);
}

void DigitalDistinguish::Backward(Sample &sample, vector<TnLayer *> &gradients)
{
    InverseTrans(sample); // 反向传播：变换梯度
    for (size_t i = 0; i < sample.m_activeLayers.size(); i++)
    {
        if (i == 0)
        { // 输入层
            for (size_t r = 0; r < gradients[i]->matrix.size(); r++)
            {
                switch (sample.m_activeLayers[i].activeFunc)
                {
                case ActiveFunc::ReLU:
                    gradients[i]->bias[r] += sample.m_activeLayers[i].a[r];
                    if (sample.m_activeLayers[i].z[r] > 0)
                    {
                        for (size_t c = 0; c < gradients[i]->matrix[r].size(); c++)
                        {
                            gradients[i]->matrix[r][c] += sample.m_activeLayers[i].a[r] * sample[c];
                        }
                    }
                    break;
                case ActiveFunc::SoftMax:
                    for (size_t c = 0; c < gradients[i]->matrix[r].size(); c++)
                    {
                        gradients[i]->matrix[r][c] += sample.m_activeLayers[i].a[r] * sample[c];
                    }
                    gradients[i]->bias[r] += sample.m_activeLayers[i].a[r];
                    break;
                default:
                    break;
                }
            }
        }
        else
        {
            for (size_t r = 0; r < gradients[i]->matrix.size(); r++)
            {
                switch (sample.m_activeLayers[i].activeFunc)
                {
                case ActiveFunc::ReLU:
                    gradients[i]->bias[r] += sample.m_activeLayers[i].a[r];
                    if (sample.m_activeLayers[i].z[r] > 0)
                    {
                        for (size_t c = 0; c < gradients[i]->matrix[r].size(); c++)
                        {
                            gradients[i]->matrix[r][c] +=
                                sample.m_activeLayers[i].a[r] * sample.m_activeLayers[i - 1].a[c];
                        }
                    }
                    break;
                case ActiveFunc::SoftMax:
                    for (size_t c = 0; c < gradients[i]->matrix[r].size(); c++)
                    {
                        gradients[i]->matrix[r][c] +=
                            sample.m_activeLayers[i].a[r] * sample.m_activeLayers[i - 1].a[c];
                    }
                    gradients[i]->bias[r] += sample.m_activeLayers[i].a[r];
                    break;
                default:
                    break;
                }
            }
        }
    }
    sample.m_activeLayers.clear();
}

void DigitalDistinguish::UpdateWeights(const vector<TnLayer *> &gradients, size_t batchSize,
                                       double stepRate)
{
    for (size_t i = 0; i < m_layers.size(); i++)
    {
        for (size_t r = 0; r < gradients[i]->matrix.size(); r++)
        {
            gradients[i]->bias[r] /= batchSize;
            m_layers[i]->bias[r] -= stepRate * gradients[i]->bias[r];
            for (size_t c = 0; c < gradients[i]->matrix[r].size(); c++)
            {
                gradients[i]->matrix[r][c] /= batchSize;
                m_layers[i]->matrix[r][c] -= stepRate * gradients[i]->matrix[r][c];
            }
        }
    }
}

DigitalDistinguish::DigitalDistinguish()
{
}

DigitalDistinguish::~DigitalDistinguish()
{
    for (auto layer : m_layers)
    {
        delete layer;
    }
}
