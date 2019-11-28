#include <vector>
class NeuralMatrix;
class Sample;

class DigitalDistinguish {
public:

	DigitalDistinguish();
	~DigitalDistinguish();

	void PushLayer(unsigned int row, unsigned int colum, float bias);
	void StartTraining(const std::vector<Sample*>& data, int sampleSize = 100);
	void ForwardPass(Sample & sample);
	void InverseTrans(Sample& sample);
	int Distinguish(Sample& sample);
	void Test(const std::vector<Sample*>& data);
	void BackwardsPass(const std::vector<Sample*>& samples, const std::vector<size_t> & indeces, double lRate, double averageCostVal);
private:
	std::vector<NeuralMatrix*> layers;
};