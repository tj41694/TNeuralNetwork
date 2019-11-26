#include "NumDistinguish.h"
#include "NeuralMatrix.h"
#include "Shuffle.h"
#include "Sample.h"

using namespace std;

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, float bias = 0) {
	NeuralMatrix* layer = new NeuralMatrix(row, colum, bias);
	layers.emplace_back(layer);
}

void DigitalDistinguish::StartTraining(const std::vector<Sample*>& samples, int sampleSize) {
	Shuffle shuff(samples.size());
	while (true) {
		const vector<size_t>& randomIndeces = shuff.GetShuffledData(sampleSize); //��ȡָ�������������������
		double sampleTotalVal = 0;
		for (size_t i = 0; i < randomIndeces.size(); i++) {
			ForwardPass(*samples[randomIndeces[i]]);
			double cost = samples[randomIndeces[i]]->GetCostValue(CostFunc::CrossEntropy);
			sampleTotalVal += cost;
		}
		double average = sampleTotalVal / sampleSize; //������ֵ
		GradiantDecent(0.01f, average);
	}
}

//���򴫵ݣ������һ��ʹ��SoftMax������⣬������ʹ��Relu
void DigitalDistinguish::ForwardPass(Sample& sample) {
	for (size_t i = 0; i < layers.size() - 1; i++) {
		sample.MatrixMultiply(*layers[i], ActiveFunc::ReLU);
	}
	sample.MatrixMultiply(*layers[layers.size() - 1], ActiveFunc::SoftMax);
}

void DigitalDistinguish::GradiantDecent(double lRate, double averageCostVal) {

}

DigitalDistinguish::DigitalDistinguish() {}

DigitalDistinguish::~DigitalDistinguish() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}