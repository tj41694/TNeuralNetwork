#include "vector"
class NeuralLayer;
class Sample;
//���ۺ������㷽ʽ
enum class CostFunc {
	//������
	MeanSquare,
	//������
	CrossEntropy
};

enum class ActiveFunc {
	Linear, //����
	ReLU  //
};

class DigitalDistinguish {
public:
	~DigitalDistinguish();
	void StartTraining(const std::vector<Sample*>& data);
	void GradiantDecent(float lRate, float costVal);
	void PushLayer(unsigned int row, unsigned int colum, float bias);
	float GetCostValue(const Sample* elem, ActiveFunc activeType, CostFunc costType);
private:
	std::vector<NeuralLayer*> layers;
};

