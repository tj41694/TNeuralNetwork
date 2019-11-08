#include "vector"
class NeuralLayer;
class Layer;

enum class CostFunc {
	MeanSquare, //均方差
	CrossEntropy //交叉熵
};

enum class ActiveFunc {
	Linear, //线性
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

