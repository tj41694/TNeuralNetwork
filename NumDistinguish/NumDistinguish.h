#include <vector>
using namespace std;
class NeuralMatrix;
class Sample;

class DigitalDistinguish
{
  public:
    DigitalDistinguish();
    ~DigitalDistinguish();

    void PushLayer(unsigned int row, unsigned int colum);
    void StartTraining(const vector<Sample *> &data, int sampleSize = 100);
    void ForwardPass(Sample &sample);
    void InverseTrans(Sample &sample);
    int Distinguish(Sample &sample);
    void Test(const vector<Sample *> &data);
    void BackwardsPass(const vector<Sample *> &samples, const vector<size_t> &indeces,
                       double lRate, double averageCostVal);

  private:
    vector<NeuralMatrix *> layers;
};