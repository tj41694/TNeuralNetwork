#include "Sample.h"
#include "TnRandom.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <utility>

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

double Sample::GetCostValue(CostFunc func, const TnVector &output) const
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

bool Sample::SaveAsBmp(const char *path) const
{
    int side = (int) sqrt((double) size());
    if (side <= 0 || side * side != (int) size())
        return false;

    int rowBytes = side * 3;
    int padding = (4 - (rowBytes % 4)) % 4;
    int pixelDataSize = (rowBytes + padding) * side;
    int fileSize = 54 + pixelDataSize;

#if defined(_MSC_VER)
    FILE *fp = nullptr;
    if (fopen_s(&fp, path, "wb") != 0)
        return false;
#else
    FILE *fp = fopen(path, "wb");
    if (fp == nullptr)
        return false;
#endif

    unsigned char header[54] = {0};
    header[0] = 'B';
    header[1] = 'M';
    header[2] = (unsigned char) (fileSize);
    header[3] = (unsigned char) (fileSize >> 8);
    header[4] = (unsigned char) (fileSize >> 16);
    header[5] = (unsigned char) (fileSize >> 24);
    header[10] = 54;
    header[14] = 40;
    header[18] = (unsigned char) (side);
    header[19] = (unsigned char) (side >> 8);
    header[20] = (unsigned char) (side >> 16);
    header[21] = (unsigned char) (side >> 24);
    header[22] = (unsigned char) (side);
    header[23] = (unsigned char) (side >> 8);
    header[24] = (unsigned char) (side >> 16);
    header[25] = (unsigned char) (side >> 24);
    header[26] = 1;
    header[28] = 24;
    header[34] = (unsigned char) (pixelDataSize);
    header[35] = (unsigned char) (pixelDataSize >> 8);
    header[36] = (unsigned char) (pixelDataSize >> 16);
    header[37] = (unsigned char) (pixelDataSize >> 24);
    fwrite(header, 1, sizeof(header), fp);

    unsigned char pad[3] = {0, 0, 0};
    for (int y = side - 1; y >= 0; y--)
    {
        for (int x = 0; x < side; x++)
        {
            double v = (*this)[y * side + x];
            v = std::max(v, 0.0);
            v = std::min(v, 1.0);
            unsigned char pixel = (unsigned char) (v * 255.0 + 0.5);
            unsigned char bgr[3] = {pixel, pixel, pixel};
            fwrite(bgr, 1, sizeof(bgr), fp);
        }
        if (padding > 0)
            fwrite(pad, 1, padding, fp);
    }
    fclose(fp);
    return true;
}

TnLayer::TnLayer(int row_, int colum_, ActiveFuncPtr actFunc, DerivFuncPtr derivFunc_)
    : activeFunc(actFunc), derivFunc(derivFunc_)
{
    // 随机源统一走 TnRandom，种子由外部指定并记录在 meta.json 里，保证可复现.
    // He/Kaiming 初始化：权重 ~ N(0, sqrt(2/fan_in))，fan_in 为该层输入维度，适配 ReLU；
    // 偏置置 0（He 的常规做法，避免偏置压过被缩小的权重）.
    const double fanIn = static_cast<double>(colum_ > 0 ? colum_ : 1);
    const double stddev = sqrt(2.0 / fanIn);
    bias.resize(row_, 0);
    for (int r = 0; r < row_; r++)
    {
        vector<double> row;
        row.resize(colum_);
        for (int c = 0; c < colum_; c++)
        {
            row[c] = RandomNormal(0.0, stddev);
        }
        matrix.emplace_back(row);
    }
}

TnLayer::TnLayer(const TnLayer &neural) : activeFunc(neural.activeFunc), derivFunc(neural.derivFunc)
{
    bias.resize(neural.bias.size(), 0);
    for (int r = 0; r < neural.row(); r++)
    {
        vector<double> row;
        row.resize(neural.col(), 0);
        matrix.emplace_back(row);
    }
}
TnLayer::TnLayer(TnLayer &&neural) noexcept
    : matrix(std::move(neural.matrix)), bias(std::move(neural.bias)),
      activeFunc(neural.activeFunc), derivFunc(neural.derivFunc), values(std::move(neural.values)),
      preActiveValues(std::move(neural.preActiveValues))
{
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
    preActiveValues = values; // 激活前保存预激活值 z
    activeFunc(values);
}

void TnLayer::operator*=(double scalar)
{
    for (size_t r = 0; r < matrix.size(); ++r)
    {
        bias[r] *= scalar;
        for (size_t c = 0; c < matrix[r].size(); ++c)
        {
            matrix[r][c] *= scalar;
        }
    }
}

TnLayer TnLayer::operator*(double scalar) const
{
    TnLayer result(*this); // 拷贝构造只复制结构与激活函数，矩阵/偏置被零初始化.
    for (size_t r = 0; r < matrix.size(); ++r)
    {
        result.bias[r] = bias[r] * scalar;
        for (size_t c = 0; c < matrix[r].size(); ++c)
        {
            result.matrix[r][c] = matrix[r][c] * scalar;
        }
    }
    return result;
}

void TnLayer::operator-=(const TnLayer &other)
{
    assert(matrix.size() == other.matrix.size());
    assert(bias.size() == other.bias.size());
    for (size_t r = 0; r < matrix.size(); ++r)
    {
        assert(matrix[r].size() == other.matrix[r].size());
        bias[r] -= other.bias[r];
        for (size_t c = 0; c < matrix[r].size(); ++c)
        {
            matrix[r][c] -= other.matrix[r][c];
        }
    }
}

void TnLayer::AddScaled(const TnLayer &other, double scale)
{
    assert(matrix.size() == other.matrix.size());
    assert(bias.size() == other.bias.size());
    for (size_t r = 0; r < matrix.size(); ++r)
    {
        assert(matrix[r].size() == other.matrix[r].size());
        bias[r] += scale * other.bias[r];
        for (size_t c = 0; c < matrix[r].size(); ++c)
        {
            matrix[r][c] += scale * other.matrix[r][c];
        }
    }
}

void TnLayer::AddScaledNormalized(const TnLayer &moment, const TnLayer &secondMoment, double lr,
                                  double eps)
{
    assert(matrix.size() == moment.matrix.size());
    assert(matrix.size() == secondMoment.matrix.size());
    assert(bias.size() == moment.bias.size());
    assert(bias.size() == secondMoment.bias.size());
    for (size_t r = 0; r < matrix.size(); ++r)
    {
        assert(matrix[r].size() == moment.matrix[r].size());
        assert(matrix[r].size() == secondMoment.matrix[r].size());
        bias[r] += lr * moment.bias[r] / (sqrt(secondMoment.bias[r]) + eps);
        for (size_t c = 0; c < matrix[r].size(); ++c)
        {
            matrix[r][c] +=
                lr * moment.matrix[r][c] / (sqrt(secondMoment.matrix[r][c]) + eps);
        }
    }
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

const vector<vector<double>> &TnLayer::Matrix() const
{
    return matrix;
}

const vector<double> &TnLayer::Bias() const
{
    return bias;
}

const TnVector &TnLayer::PreActiveValues() const
{
    return preActiveValues;
}

double TnLayer::NormSquared() const
{
    double sum = 0;
    for (const auto &row : matrix)
    {
        for (double v : row)
        {
            sum += v * v;
        }
    }
    for (double v : bias)
    {
        sum += v * v;
    }
    return sum;
}

TnLayer TnLayer::Square() const
{
    TnLayer result(*this); // 拷贝构造只复制结构与激活函数，矩阵/偏置被零初始化，不消耗随机源.
    for (int r = 0; r < row(); ++r)
    {
        result.bias[r] = bias[r] * bias[r];
        for (int c = 0; c < col(); ++c)
        {
            result.matrix[r][c] = matrix[r][c] * matrix[r][c];
        }
    }
    return result;
}

TnLayer TnLayer::Clone() const
{
    TnLayer result(*this);
    result.matrix = matrix;
    result.bias = bias;
    return result;
}

void TnLayer::CalcGradient(const TnVector &prevLyActiveValues, const TnLayer &curLayer,
                           TnVector *preGradients)
{
    derivFunc(curLayer.preActiveValues, values);
    if (preGradients != nullptr)
    {
        preGradients->clear();
        preGradients->resize(prevLyActiveValues.size(), 0);
    }
    for (int i = 0; i < (int) values.size(); ++i)
    {
        bias[i] += values[i]; // 偏移量的偏导数为常量1

        assert(prevLyActiveValues.size() == curLayer.matrix[i].size());
        for (int j = 0; j < (int) prevLyActiveValues.size(); ++j)
        {
            if (preGradients != nullptr)
            {
                // 前一层的激活值偏导梯度为此层权重累加
                (*preGradients)[j] += (curLayer.matrix[i][j] * values[i]);
            }
            // 权重矩阵的偏导梯度值为上一层的激活值累加
            matrix[i][j] += (prevLyActiveValues[j] * values[i]);
        }
    }
}
