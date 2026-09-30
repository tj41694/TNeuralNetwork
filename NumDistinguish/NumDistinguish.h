#include "Sample.h"
#include <vector>
using namespace std;

void Linear(TnVector &vec);
void Sigmoid(TnVector &vec);
void ReLU(TnVector &vec);
void SoftMax(TnVector &vec);

class DigitalDistinguish
{
  public:
    DigitalDistinguish();
    ~DigitalDistinguish();

    void PushLayer(unsigned int row, unsigned int colum, ActiveFuncPtr activeFunc);
    void Training(const vector<Sample *> &data, int sampleSize = 100);
    int Distinguish(Sample &sample);
    void Validate(const vector<Sample *> &data);

  private:
    TnVector ForwardPass(Sample &sample);
    void InverseTrans(Sample &sample) const;
    void Backward(Sample &sample, vector<TnLayer *> &gradients);
    void UpdateWeights(const vector<TnLayer *> &gradients, size_t batchSize, double stepRate);

  private:
    vector<TnLayer *> m_layers;
};