#include "Sample.h"
#include "NeuralMatrix.h"

using namespace std;
Sample::Sample()
{
    m_realValue = 0;
}

Sample::Sample(const Sample &sample_)
{
    m_data = sample_.m_data;
    m_realValue = sample_.m_realValue;
}

Sample::Sample(int num_, const float *data, unsigned int floatCount)
{
    m_data.resize(floatCount);
    for (unsigned int i = 0; i < floatCount; i++)
    {
        m_data[i] = data[i];
    }
    m_realValue = num_;
}

void Sample::MatrixMultiply(const NeuralMatrix &neuralMat, ActiveFunc func)
{

    vector<double> *lastActiveLayer;

    m_activeLayers.size() == 0 ? lastActiveLayer = &m_data
                             : lastActiveLayer = &m_activeLayers[m_activeLayers.size() - 1].out;

    if ((int) lastActiveLayer->size() != neuralMat.column)
    {
        printf("err.. Dimension not match..\n");
        return;
    }

    SampleLayer layer;
    layer.activeFunc = func;

    layer.net.resize(neuralMat.row);
    layer.out.resize(neuralMat.row);
    for (int r = 0; r < neuralMat.row; r++)
    { // 计算输出（权重×输入）
        double val = 0;
        for (int c = 0; c < neuralMat.column; c++)
        {
            val += (*lastActiveLayer)[c] * neuralMat.matrix[r][c];
        }
        layer.net[r] = val + neuralMat.bias[r];
    }
    switch (layer.activeFunc)
    {
    case ActiveFunc::ReLU:
        for (unsigned int i = 0; i < layer.net.size(); i++)
        {
            layer.net[i] > 0 ? layer.out[i] = layer.net[i] : layer.out[i] = 0;
        }
        break;
    case ActiveFunc::SoftMax:
    {
        double total = 0; // 求和
        for (unsigned int i = 0; i < layer.net.size(); i++)
        {
            // printf("ori: %f\n", layer.net.activation[i]);
            layer.out[i] = exp(layer.net[i]);
            total += layer.out[i];
        }
        for (unsigned int i = 0; i < layer.net.size(); i++)
        { // 归一化
            layer.out[i] = layer.out[i] / total;
            // printf("softmax: %f\n", layer.out.activation[i]);
        }
    }
    break;
    case ActiveFunc::Linear:
    default:
        for (unsigned int i = 0; i < layer.net.size(); i++)
        {
            layer.out[i] = layer.net[i];
        }
        break;
    }
    m_activeLayers.push_back(layer);
}

double Sample::GetCostValue(CostFunc func)
{
    if (!costValValid)
    {
        const vector<double> &outputLayer = m_activeLayers[m_activeLayers.size() - 1].out;
        switch (func)
        {
        case CostFunc::CrossEntropy:
            costVal = -log(outputLayer[m_realValue]);
            break;
        case CostFunc::MeanSquare:
        default:
            for (int i = 0; i < (int) outputLayer.size(); i++)
            {
                if (i == m_realValue)
                {
                    costVal += (outputLayer[i] - 1.0) * (outputLayer[i] - 1.0);
                }
                else
                {
                    costVal += outputLayer[i] * (double) outputLayer[i];
                }
            }
            break;
        }
        costValValid = true;
    }
    return costVal;
}

Sample &Sample::operator=(const Sample &sample_)
{
    if (this == &sample_)
    {
        return *this;
    }
    m_data = sample_.m_data;
    m_activeLayers = sample_.m_activeLayers;
    m_realValue = sample_.m_realValue;
    return *this;
}

Sample::~Sample()
{
}