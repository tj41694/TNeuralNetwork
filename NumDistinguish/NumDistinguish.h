#include "vector"
class NeuralLayer;
class Sample;
//代价函数计算方式
enum class CostFunc {
	//均方差
	MeanSquare,
	//交叉熵
	CrossEntropy
};

enum class ActiveFunc {
	Linear, //线性
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

