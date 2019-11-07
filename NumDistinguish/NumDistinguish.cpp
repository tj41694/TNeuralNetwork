#include "NumDistinguish.h"
#include "NeuralLayer.h"

using namespace std;
void DigitalDistinguish::StartTraining(const std::vector<RawData>& data) {

	size_t count = data.size();
	float totalVAL = 0;
	float OneHundredVal = 0;
	unsigned int c = 0;
	vector<DataElem> reualts;
	for (size_t i = 0; i < count; i++) {
		float cost = GetCostValue((DataElem*)&data[i], ActiveFunc::ReLU, CostFunc::CrossEntropy);
		totalVAL += cost;
		OneHundredVal += cost;
		c++;
		if (i % 100 == 0 || i == count - 1) {
			float aver = OneHundredVal / c;

			GradiantDecent(0.01f, aver);
			OneHundredVal = 0;
			c = 0;
		}
	}
	printf("%f\n", totalVAL / count);
}

void DigitalDistinguish::GradiantDecent(float lRate, float costVal) {

}

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, float bias = 0) {
	NeuralLayer* layer = new NeuralLayer(row, colum, bias);
	layers.emplace_back(layer);
}

float DigitalDistinguish::GetCostValue(const DataElem* elem, ActiveFunc activeType, CostFunc costType) {
	DataElem tmpLayer = *elem;
	for (unsigned int i = 0; i < layers.size(); i++) {
		DataElem hiddenLater = layers[i]->Multiply(tmpLayer.pixs, tmpLayer.size);
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
	float costVal = 0;
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
				costVal += (tmpLayer.pixs[i] - 1) * (tmpLayer.pixs[i] - 1);
			}
			else {
				costVal += tmpLayer.pixs[i] * tmpLayer.pixs[i];
			}
		}
		break;
	}
	return costVal;
}

DigitalDistinguish::~DigitalDistinguish() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}