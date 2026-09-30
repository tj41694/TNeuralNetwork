#pragma once
#include <math.h>
#include <vector>

using namespace std;

// 代价函数计算方式
enum class CostFunc
{
    MeanSquare, // 均方差
    CrossEntropy // 交叉熵
};

class TnVector : public vector<double>
{
};
using ActiveFuncPtr = void (*)(TnVector &vec);
using DerivFuncPtr = void (*)(const TnVector &preActiveValues, TnVector &vec);

class TnLayer
{
  public:
    TnLayer(int row_, int colum_, ActiveFuncPtr actFunc, DerivFuncPtr derivFunc_);
    TnLayer(const TnLayer &neural);
    int row() const;
    int col() const;
    void operator*=(const vector<double> &vec);
    const TnVector &Values() const;
    TnVector &Values();
    void CalcGradient(const TnVector &preActiveValues, const vector<vector<double>> &curMatrix,
                      TnVector *preGradients);

  private:
  public:
    vector<vector<double>> matrix;
    vector<double> bias;

  private:
    ActiveFuncPtr activeFunc = nullptr;
    DerivFuncPtr derivFunc = nullptr;
    // 在前向传播里，此值代表当前层的激活值；在反向传播里，此值代表当前层临时计算出的梯度.
    TnVector values;
};

class Sample : public TnVector
{
  public:
    int m_realValue;

  public:
    Sample(const Sample &);
    Sample(int num_, const float *data, unsigned int floatCount);

    double GetCostValue(CostFunc func, const TnVector & output) const;
};
