#include "vector"
class NeuralMatrix;
class Sample;

class DigitalDistinguish {
public:

	DigitalDistinguish();
	~DigitalDistinguish();

	void PushLayer(unsigned int row, unsigned int colum, float bias);
	void StartTraining(const std::vector<Sample*>& data, int sampleSize = 100);
	void ForwardPass(Sample & sample);
	void GradiantDecent(double lRate, double costVal);
private:
	std::vector<NeuralMatrix*> layers;
};

