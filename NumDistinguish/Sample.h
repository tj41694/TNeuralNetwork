#pragma once
#include <math.h>
#include <vector>

using namespace std;

enum class ActiveFunc
{
    Sigmoid,
    ReLU,
    SoftMax
};

// 代价函数计算方式
enum class CostFunc
{
    MeanSquare, // 均方差
    CrossEntropy // 交叉熵
};

class TnVector;

using ActiveFuncPtr = void (*)(TnVector &vec);

class TnLayer
{
  public:
    TnLayer(int row_, int colum_, ActiveFuncPtr actFunc);
    TnLayer(const TnLayer &neural);
    int row() const;
    int col() const;
    void Active(TnVector & vec) const;

  public:
    vector<vector<double>> matrix;
    vector<double> bias;

  private:
    ActiveFuncPtr activeFunc = nullptr;
};

class TnVector : public vector<double>
{
  public:
    TnVector operator*(const TnLayer &matrix) const;
};

struct SampleLayer
{
    TnVector z;
    TnVector a;
    ActiveFunc activeFunc;
};

class Sample : public TnVector
{
  public:
    int m_realValue;
    vector<SampleLayer> m_activeLayers;

  public:
    Sample();
    Sample(const Sample &);
    Sample(int num_, const float *data, unsigned int floatCount);
    Sample operator*(const TnLayer &matrix) const;

    void MatrixMultiply(const TnLayer &layer, ActiveFunc func);
    double GetCostValue(CostFunc func);
};
