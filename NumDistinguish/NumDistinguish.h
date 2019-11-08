#include "vector"
class NeuralLayer;
class Layer;

enum class CostFunc {
	MeanSquare, //������
	CrossEntropy //������
};

enum class ActiveFunc {
	Linear, //����
	ReLU  //
};

class DigitalDistinguish {
public:
	~DigitalDistinguish();
	void StartTraining(const std::vector<Layer>& data);
	void GradiantDecent(float lRate, float costVal);
	void PushLayer(unsigned int row, unsigned int colum, float bias);
	float GetCostValue(const Layer& elem, ActiveFunc activeType, CostFunc costType);
private:
	std::vector<NeuralLayer*> layers;
};

