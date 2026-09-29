#pragma once
#include <math.h>
#include <vector>

using namespace std;
class NeuralMatrix;

enum class ActiveFunc
{
    Linear,
    Sigmoid,
    ReLU,
    SoftMax
};

// 代价函数计算方式
enum class CostFunc
{
    // 均方差
    MeanSquare,
    // 交叉熵
    CrossEntropy
};

struct SampleLayer
{
    vector<double> net;
    vector<double> out;
    ActiveFunc activeFunc;
};

class Sample
{
  public:
    int m_realValue;
    vector<double> m_data;
    vector<SampleLayer> m_activeLayers;

  public:
    Sample();
    Sample(const Sample &);
    Sample(int num_, const float *data, unsigned int floatCount);
    ~Sample();

    void MatrixMultiply(const NeuralMatrix &layer, ActiveFunc func);
    double GetCostValue(CostFunc func);

    Sample &operator=(const Sample &ly);

  private:
    double costVal = 0;
    bool costValValid = false;
};
