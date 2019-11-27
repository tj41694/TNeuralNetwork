#include "NumDistinguish.h"
#include "NeuralMatrix.h"
#include "Shuffle.h"
#include "Sample.h"

using namespace std;

void DigitalDistinguish::PushLayer(unsigned int row, unsigned int colum, float bias = 0) {
	NeuralMatrix* layer = new NeuralMatrix(row, colum, bias);
	layers.emplace_back(layer);
}

void DigitalDistinguish::StartTraining(const vector<Sample*>& samples, int sampleSize) {
	Shuffle shuff(samples.size());
	double averageCost = 100000.0;
	while (averageCost > 0.5) {
		const vector<size_t>& randomIndeces = shuff.GetShuffledData(sampleSize); //获取指定数量的随机样本索引
		double sampleTotalVal = 0;
		for (size_t i = 0; i < randomIndeces.size(); i++) {
			ForwardPass(*samples[randomIndeces[i]]);
			double cost = samples[randomIndeces[i]]->GetCostValue(CostFunc::CrossEntropy);
			sampleTotalVal += cost;
		}
		averageCost = sampleTotalVal / sampleSize; //样本均值
		BackwardsPass(samples, randomIndeces, 0.05f, averageCost);
	}
}

void DigitalDistinguish::ForwardPass(Sample& sample) {
	for (size_t i = 0; i < layers.size() - 1; i++) {
		sample.MatrixMultiply(*layers[i], ActiveFunc::ReLU);
	}
	sample.MatrixMultiply(*layers[layers.size() - 1], ActiveFunc::SoftMax);
}

void DigitalDistinguish::BackwardsPass(const vector<Sample*>& samples, const vector<size_t>& indeces, double lRate, double averageCostVal) {
	for (long long i = layers.size() - 1; i > -1; i--) {
		NeuralMatrix gradient(*layers[i], true);
		if (i != 0) {
			for (auto index : indeces) {
				const Sample& sample = *samples[index];
				gradient.bias += (sample.activeLayers[i].out[sample.trueValue] - 1);
				for (int r = 0; r < gradient.matrix.size(); r++) {
					for (int c = 0; c < gradient.matrix[r].size(); c++) {
						if (r == sample.trueValue) {
							gradient.matrix[r][c] = (sample.activeLayers[i].out[r] - 1) * sample.activeLayers[i - 1].out[r];
						}
						else {
							gradient.matrix[r][c] = sample.activeLayers[i].out[r] * sample.activeLayers[i - 1].out[r];
						}
					}
				}
			}
		}
		else {

		}
		gradient.bias /= indeces.size();
	}
}

DigitalDistinguish::DigitalDistinguish() {

}

DigitalDistinguish::~DigitalDistinguish() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}