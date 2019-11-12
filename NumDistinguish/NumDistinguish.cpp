#include "NumDistinguish.h"
#include "NeuralLayer.h"
#include "Shuffle.h"
#include "Layer.h"

using namespace std;
void DigitalDistinguish::StartTraining(const std::vector<Layer*>& data) {

	vector<Layer> reualts;
	int sampleSize = 100;
	Shuffle shuff(data);
	while (true) {
		vector<unsigned int>* sample = shuff.GetShuffledData(sampleSize); //样本索引
		float sampleTotalVal = 0;
		for (size_t i = 0; i < sample->size(); i++) {
			unsigned int index = (*sample)[i];
			float cost = GetCostValue(data[index], ActiveFunc::ReLU, CostFunc::CrossEntropy);
			sampleTotalVal += cost;
		}
		delete sample;
		float average = sampleTotalVal / sampleSize; //样本均值
		GradiantDecent(0.01f, average);
	}
}

void DigitalDistinguish::GradiantDecent(float lRate, float averageCostVal) {

}

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, float bias = 0) {
	NeuralLayer* layer = new NeuralLayer(row, colum, bias);
	layers.emplace_back(layer);
}

float DigitalDistinguish::GetCostValue(const Layer* elem, ActiveFunc activeType, CostFunc costType) {

	Layer* tmpLayer = elem;
	for (unsigned int i = 0; i < layers.size(); i++) {
		Layer* hiddenLater = layers[i]->MatrixMultiply(tmpLayer.activation, tmpLayer.size);

		if (i != layers.size() - 1) {
			switch (activeType) {
			case ActiveFunc::ReLU:
				hiddenLater->Relu(); //非输出层使用relu作为激活函数
				break;
			case ActiveFunc::Linear:
			default:
				break;
			}
		}
		tmpLayer = hiddenLater;
	}
	double costVal = 0;
	switch (costType) {
	case CostFunc::CrossEntropy:
	{
		Layer softMax = tmpLayer.SoftMax();
		costVal = -log(softMax.activation[elem.trueValue]);
		delete[] softMax.activation;
	}
	break;
	case CostFunc::MeanSquare:
	default:
		for (unsigned int i = 0; i < tmpLayer.size; i++) {
			if (i == elem.trueValue - 1) {
				costVal += (tmpLayer.activation[i] - 1.0) * (tmpLayer.activation[i] - 1.0);
			}
			else {
				costVal += tmpLayer.activation[i] * (double)tmpLayer.activation[i];
			}
		}
		break;
	}
	return (float)costVal;
}

DigitalDistinguish::~DigitalDistinguish() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}