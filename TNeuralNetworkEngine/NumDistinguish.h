#pragma once
#include "Recorder.h"
#include "Sample.h"
#include <atomic>
#include <cstdint>
#include <functional>
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
    uint32_t lrDecayFromStep = 17500;
    // 为空则不做任何遥测记录
    Recorder *recorder = nullptr;
    // 参与激活快照的固定样本；为空则不记录激活
    const vector<Sample *> *probes = nullptr;
    // 周期性评估用的验证集（通常是测试集）；为空则不做周期性准确率评估
    const vector<Sample *> *validationData = nullptr;
    // 每隔多少步在独立线程上对训练集与验证集各评估一次；<=0 或没有 validationData 则关闭
    uint32_t metricsEvery = 100;
    // 置位后在当前 step 结束时正常收尾
    const std::atomic<bool> *stopRequested = nullptr;
};

class DigitalDistinguish
{
  public:
    ~DigitalDistinguish();

    void PushLayer(int input, int output, ActiveFuncPtr activeFunc, DerivFuncPtr derivFunc);
    void Training(const vector<Sample *> &samples, const TrainingOptions &options);
    void TrainingAdam(const vector<Sample *> &samples, const TrainingOptions &options,
                      double momentumBeta, double rsmBeta);
    int Distinguish(const Sample &sample);
    // 在 data 上评估，返回准确率（0~100），同时打印到控制台.
    double Validate(const vector<Sample *> &data);

  private:
    void Forward(const Sample &input);
    void Backward(const Sample &input, const TnVector &output,
                  vector<TnLayer *> &gradientLayers, size_t batchSize) const;
    void UpdateWeights(const vector<TnLayer *> &gradients, double stepRate);
    void UpdateWeightsAdam(double inputLr, uint32_t step, double momentumBeta,
                                      double rsmBeta);
    // 融合 m_adamGradients（成员变量里的动量）与当前 batch 计算出的梯度：m = β·m + (1−β)·g.
    // 偏差修正不在这里做，由调用方在更新时按 1/(1−β^t) 处理，避免修正值被反复复用而放大.
    void FuseGradients(const vector<TnLayer *> &currentGradients, double momentumRatio,
                       double rsmRatio);
    // 初始化 Adam 的跨 step 状态：一阶动量与逐参数学习率，结构对齐 m_layers，数值零初始化.
    void InitAdamState();
    // 两种训练方式共用的循环：采样 batch、前向、反向、遥测、周期评估、打印。
    // optimize 负责本步优化器逻辑：消费 gradients、更新权重，并回填用于日志的 lr 与 gradNorm。
    void RunTrainingLoop(const vector<Sample *> &samples, const TrainingOptions &options,
                         const function<void(uint32_t step, vector<TnLayer *> &gradients, double &lr,
                                             double &gradNorm)> &optimize);

  private:
    vector<TnLayer *> m_layers;
    vector<TnLayer *> m_adamGradients;
    vector<TnLayer *> m_adamLearningRates;
};
