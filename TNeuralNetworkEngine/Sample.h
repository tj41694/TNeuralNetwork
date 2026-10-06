#pragma once
#include <math.h>
#include <vector>

using namespace std;

// 代价函数计算方式
enum class CostFunc
{
    MeanSquare,  // 均方差
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
    TnLayer(TnLayer &&neural) noexcept;
    int row() const;
    int col() const;
    void operator*=(const vector<double> &vec);
    TnLayer operator*(double scalar) const;
    void operator*=(double scalar);
    void operator-=(const TnLayer &other);
    // this += scale * other，逐元素作用于权重矩阵与偏置.
    void AddScaled(const TnLayer &other, double scale);
    // Adam 逐参数更新：this += scale * moment / (sqrt(secondMoment) + eps).
    void AddScaledNormalized(const TnLayer &moment, const TnLayer &secondMoment, double lr,
                             double eps);
    const TnVector &Values() const;
    TnVector &Values();
    void CalcGradient(const TnVector &prevLyActiveValues, const TnLayer &curLayer,
                           TnVector *preGradients);

    // 只读访问器：供遥测读取权重与预激活值，不影响训练逻辑.
    const vector<vector<double>> &Matrix() const;
    const vector<double> &Bias() const;
    const TnVector &PreActiveValues() const;

    // 权重矩阵与偏置的平方和（Frobenius 范数的平方），用于遥测统计梯度/权重的范数.
    double NormSquared() const;

    // 返回一个新层，其权重矩阵与偏置均为当前层对应值的平方.
    TnLayer Square() const;

    // 深拷贝权重矩阵、偏置与激活函数，得到一个结构与数值都相同的新层.
    // 用于评估线程在独立网络上做前向，避免读写训练网络的共享状态.
    TnLayer Clone() const;

  protected:
    vector<vector<double>> matrix;
    vector<double> bias;

  private:
    ActiveFuncPtr activeFunc = nullptr;
    DerivFuncPtr derivFunc = nullptr;
    // 在前向传播里，此值代表当前层的激活值；在反向传播里，此值代表当前层临时计算出的梯度.
    TnVector values;
    // 当前层前向传播的预激活值 z，反向传播时用于计算激活函数的导数.
    TnVector preActiveValues;
};

class Sample : public TnVector
{
  public:
    int m_realValue;

  public:
    Sample(const Sample &);
    Sample(int realNum, const float *data, unsigned int ct);

    double GetCostValue(CostFunc func, const TnVector &output) const;
    bool SaveAsBmp(const char *path) const;
};
