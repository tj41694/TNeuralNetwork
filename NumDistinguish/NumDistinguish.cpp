#include "NumDistinguish.h"
#include "NeuralLayer.h"
#include "Shuffle.h"
#include "Sample.h"

using namespace std;

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, float bias = 0) {
	NeuralLayer* layer = new NeuralLayer(row, colum, bias);
	layers.emplace_back(layer);
}

void DigitalDistinguish::StartTraining(const std::vector<Sample*>& data, int sampleSize) {
	vector<Sample> reualts;
	Shuffle shuff(data);
	while (true) {
		const vector<unsigned int> & randomIndeces = shuff.GetShuffledData(sampleSize); //获取指定数量的随机样本索引
		float sampleTotalVal = 0;
		for (size_t i = 0; i < randomIndeces.size(); i++) {
			unsigned int index = randomIndeces[i];
			float cost = GetCostValue(data[index], ActiveFunc::ReLU, CostFunc::CrossEntropy);
			sampleTotalVal += cost;
		}
		float average = sampleTotalVal / sampleSize; //样本均值
		GradiantDecent(0.01f, average);
	}
}

void DigitalDistinguish::GradiantDecent(float lRate, float averageCostVal) {

}

float DigitalDistinguish::GetCostValue(const Sample* sample, ActiveFunc activeType, CostFunc costType) {
	//第1层
	Sample* hdLayer = layers[0]->MatrixMultiply(*sample);
	hdLayer->Relu();
	//第2到n层
	for (unsigned int i = 1; i < layers.size(); i++) {
		Sample* tempLater = layers[i]->MatrixMultiply(*hdLayer);
		if (i != layers.size() - 1) {
			switch (activeType) {
			case ActiveFunc::ReLU:
				tempLater->Relu(); //非输出层使用relu作为激活函数
				break;
			case ActiveFunc::Linear:
			default:
				break;
			}
		}
		delete hdLayer;
		hdLayer = tempLater;
	}
	double costVal = 0;
	switch (costType) {
	case CostFunc::CrossEntropy:
	{
		Sample softMax = hdLayer->SoftMax(); //调用复制构造函数
		costVal = -log(softMax.activation[sample->trueValue]);
	}
	break;
	case CostFunc::MeanSquare:
	default:
		for (unsigned int i = 0; i < hdLayer->size; i++) {
			if (i == sample->trueValue - 1) {
				costVal += (hdLayer->activation[i] - 1.0) * (hdLayer->activation[i] - 1.0);
			}
			else {
				costVal += hdLayer->activation[i] * (double)hdLayer->activation[i];
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