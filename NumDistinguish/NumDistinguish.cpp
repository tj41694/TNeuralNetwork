#include "NumDistinguish.h"
#include "NeuralLayer.h"
#include "Shuffle.h"

using namespace std;
void DigitalDistinguish::StartTraining(const std::vector<RawData>& data) {

	vector<DataElem> reualts;
	int sampleSize = 100;
	Shuffle shuff(data);
	while (true) {
		vector<unsigned int>* sample = shuff.GetShuffledData(sampleSize); //样本索引
		float sampleTotalVal = 0;
		for (size_t i = 0; i < sample->size(); i++) {
			float cost = GetCostValue((DataElem*)&data[(*sample)[i]], ActiveFunc::ReLU, CostFunc::CrossEntropy);
			sampleTotalVal += cost;
		}
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

float DigitalDistinguish::GetCostValue(const DataElem* elem, ActiveFunc activeType, CostFunc costType) {
	DataElem tmpLayer = *elem;
	for (unsigned int i = 0; i < layers.size(); i++) {
		DataElem hiddenLater = layers[i]->MatrixMultiply(tmpLayer.pixs, tmpLayer.size);
		if (i != 0) { delete[] tmpLayer.pixs; }

		if (i != layers.size() - 1) {
			switch (activeType) {
			case ActiveFunc::ReLU:
				hiddenLater.Relu(); //非输出层使用relu作为激活函数
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
		DataElem softMax = tmpLayer.SoftMax();
		costVal = -log(softMax.pixs[((RawData*)elem)->num]);
		delete[] softMax.pixs;
	}
	break;
	case CostFunc::MeanSquare:
	default:
		for (unsigned int i = 0; i < tmpLayer.size; i++) {
			if (i == ((RawData*)elem)->num - 1) {
				costVal += (tmpLayer.pixs[i] - 1.0) * (tmpLayer.pixs[i] - 1.0);
			}
			else {
				costVal += tmpLayer.pixs[i] * (double)tmpLayer.pixs[i];
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