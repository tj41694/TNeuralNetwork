#pragma once
#include "Recorder.h"
#include "Sample.h"
#include <atomic>
#include <cstdint>
#include <vector>
using namespace std;

void Sigmoid(TnVector &input);
void DerivSigmoid(const TnVector &preActiveValues, TnVector &vec);

void ReLU(TnVector &input);
void DerivReLU(const TnVector &preActiveValues, TnVector &vec);

void SoftMax(TnVector &input);
void DerivSoftMax(const TnVector &preActiveValues, TnVector &vec);

// 一次训练的全部可调项。默认值与重构前的硬编码行为完全一致.
struct TrainingOptions
{
    int batchSize = 100;
    uint32_t totalSteps = 25000;
    double lrHigh = 0.1;
    double lrLow = 0.05;
    uint32_t lrDecayFromStep = 20000;
    // 为空则不做任何遥测记录
    Recorder *recorder = nullptr;
    // 参与激活快照的固定样本；为空则不记录激活
    const vector<Sample *> *probes = nullptr;
    // 置位后在当前 step 结束时正常收尾
    const std::atomic<bool> *stopRequested = nullptr;
};

class DigitalDistinguish
{
  public:
    ~DigitalDistinguish();

    void PushLayer(int input, int output, ActiveFuncPtr activeFunc,
                   DerivFuncPtr derivFunc);
    void Training(const vector<Sample *> &samples, const TrainingOptions &options);
    int Distinguish(const Sample &sample);
    void Validate(const vector<Sample *> &data);

  private:
    void Forward(const Sample &input);
    void Backward(const Sample &input, const TnVector &output, vector<TnLayer *> &gradients) const;
    void UpdateWeights(const vector<TnLayer *> &gradients, size_t batchSize, double stepRate);

  private:
    vector<TnLayer *> m_layers;
};
