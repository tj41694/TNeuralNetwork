#include "Sample.h"
#include <vector>
using namespace std;

void Sigmoid(TnVector &vec);
void DerivSigmoid(const TnVector &preActiveValues, TnVector &vec);

void ReLU(TnVector &vec);
void DerivReLU(const TnVector &preActiveValues, TnVector &vec);

void SoftMax(TnVector &vec);
void DerivSoftMax(const TnVector &preActiveValues, TnVector &vec);

class DigitalDistinguish
{
  public:
    ~DigitalDistinguish();

    void PushLayer(unsigned int input, unsigned int output, ActiveFuncPtr activeFunc, DerivFuncPtr derivFunc);
    void Training(const vector<Sample *> &data, int batchSize);
    int Distinguish(const Sample &sample);
    void Validate(const vector<Sample *> &data);

  private:
    void Forward(const Sample &input);
    void Backward(const Sample &input, const TnVector &output, vector<TnLayer *> &gradients) const;
    void UpdateWeights(const vector<TnLayer *> &gradients, size_t batchSize, double stepRate);

  private:
    vector<TnLayer *> m_layers;
};