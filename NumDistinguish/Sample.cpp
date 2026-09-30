#include "Sample.h"
#include <cassert>
#include <cstdlib> // Header file needed to use srand and rand
#include <ctime>   // Header file needed to use time

using namespace std;

Sample::Sample(const Sample &sample_) : TnVector(sample_)
{
    m_realValue = sample_.m_realValue;
}

Sample::Sample(int realNum, const float *data, unsigned int ct)
{
    resize(ct);
    for (unsigned int i = 0; i < ct; i++)
    {
        (*this)[i] = data[i];
    }
    m_realValue = realNum;
}

double Sample::GetCostValue(CostFunc func, const TnVector & output) const
{
    double costVal = 0;
    switch (func)
    {
    case CostFunc::CrossEntropy:
        costVal = -log(output[m_realValue]);
        break;
    case CostFunc::MeanSquare:
    default:
        for (int i = 0; i < (int) output.size(); i++)
        {
            if (i == m_realValue)
                costVal += (output[i] - 1.0) * (output[i] - 1.0);
            else
                costVal += output[i] * (double) output[i];
        }
        break;
    }
    return costVal;
}

TnLayer::TnLayer(int row_, int colum_, ActiveFuncPtr actFunc, DerivFuncPtr derivFunc_)
    : activeFunc(actFunc), derivFunc(derivFunc_)
{
    static bool initial = false;
    if (!initial)
    {
        srand((unsigned int) time(0));
        initial = true;
    }
    bias.resize(row_);
    for (int r = 0; r < row_; r++)
    {
        vector<double> row;
        row.resize(colum_);
        for (int c = 0; c < colum_; c++)
        {
            row[c] = rand() * 2.0 / RAND_MAX - 1.0;
        }
        matrix.emplace_back(row);
        bias[r] = rand() * 2.0 / RAND_MAX - 1.0;
    }
}

TnLayer::TnLayer(const TnLayer &neural)
    : activeFunc(neural.activeFunc), derivFunc(neural.derivFunc)
{
    bias.resize(neural.bias.size(), 0);
    for (int r = 0; r < neural.row(); r++)
    {
        vector<double> row;
        row.resize(neural.col(), 0);
        matrix.emplace_back(row);
    }
}
int TnLayer::row() const
{
    return (int) matrix.size();
}
int TnLayer::col() const
{
    if (matrix.empty())
        return 0;
    return (int) matrix.front().size();
}

void TnLayer::operator*=(const vector<double> &vec)
{
    assert(vec.size() == col());
    values.clear();
    values.resize(row(), 0);
    for (int r = 0; r < row(); r++)
    {
        for (int c = 0; c < col(); c++)
        {
            values[r] += vec[c] * matrix[r][c];
        }
        values[r] = values[r] + bias[r];
    }
    activeFunc(values);
}
// 在前向传播里，此值代表当前层的激活值；在反向传播里，此值代表当前层临时计算出的梯度.
const TnVector &TnLayer::Values() const
{
    return values;
}
// 在前向传播里，此值代表当前层的激活值；在反向传播里，此值代表当前层临时计算出的梯度.
TnVector &TnLayer::Values()
{
    return values;
}

void TnLayer::CalcGradient(const TnVector &preActiveValues, const vector<vector<double>> &curMatrix,
                           TnVector *preGradients)
{
    derivFunc(preActiveValues, values);
    if (preGradients)
    {
        preGradients->clear();
        preGradients->resize(preActiveValues.size(), 0);
    }
    for (int i = 0; i < (int) values.size(); ++i)
    {
        bias[i] += values[i]; // 偏移量的偏导数为常量1

        assert(preActiveValues.size() == curMatrix[i].size());
        for (int j = 0; j < (int) preActiveValues.size(); ++j)
        {
            if (preGradients)
            {
                // 前一层的激活值偏导梯度为此层权重累加
                (*preGradients)[j] += (curMatrix[i][j] * values[i]);
            }
            // 权重矩阵的偏导梯度值为上一层的激活值累加
            matrix[i][j] += (preActiveValues[j] * values[i]);
        }
    }
}
