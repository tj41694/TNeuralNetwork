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
    void Test(const vector<Sample *> &data);

  private:
    TnVector ForwardPass(Sample &sample);
    void InverseTrans(Sample &sample);
    void BackwardsPass(const vector<Sample *> &samples, const vector<size_t> &indeces, double lRate,
                       double averageCostVal);

  private:
    vector<TnLayer *> m_layers;
};