#include "vector"
class NeuralLayer;

enum class CostFunc {
	MeanSquare,
	CrossEntropy
};

enum class ActiveFunc {
	Linear,
	ReLU
};

struct DataElem {
	float* pixs;
	unsigned int size;
	void Relu() {
		for (unsigned int i = 0; i < size; i++) {
			if (pixs[i] < 0) {
				pixs[i] = 0;
			}
		}
	}
	DataElem SoftMax() {
		//先求和
		double total = 0;
		DataElem result;
		result.pixs = new float[size] { 0 };
		for (unsigned int i = 0; i < size; i++) {
			//printf("ori: %f\n", pixs[i]);
			result.pixs[i] = (float)exp(pixs[i]);
			total += result.pixs[i];
		}
		//再赋值
		for (unsigned int i = 0; i < size; i++) {
			result.pixs[i] = (float)(result.pixs[i] / total);
			//printf("softmax: %f\n", result.pixs[i]);
		}
		return result;
	}
};

struct RawData : DataElem {
	int num;
	RawData(int num_, const float* data, unsigned int dataCount) {
		size = dataCount / sizeof(float);
		pixs = new float[size];
		memcpy(pixs, data, dataCount);
		num = num_;
	}

};
class DigitalDistinguish {
public:
	~DigitalDistinguish();
	void StartTraining(const std::vector<RawData>& data);
	void GradiantDecent(float lRate, float costVal);
	void PushLayer(unsigned int row, unsigned int colum, float bias);
	float GetCostValue(const DataElem* elem, ActiveFunc activeType, CostFunc costType);
private:
	std::vector<NeuralLayer*> layers;
};

