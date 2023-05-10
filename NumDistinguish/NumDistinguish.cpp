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
	int count = sampleSize;
	int times = 0;
	while (times++ < 5000) {
		const vector<size_t>& randomIndeces = shuff.GetShuffledData(sampleSize); //获取指定数量的随机样本索引
		double sampleTotalVal = 0;
		for (size_t i = 0; i < randomIndeces.size(); i++) {
			ForwardPass(*samples[randomIndeces[i]]);
			double cost = samples[randomIndeces[i]]->GetCostValue(CostFunc::CrossEntropy);
			sampleTotalVal += cost;
		}
		averageCost = sampleTotalVal / sampleSize; //样本均值
		BackwardsPass(samples, randomIndeces, 0.15f, averageCost);
		printf("Sample Count: %d \t Cost Value: %.5f \n", count, averageCost);
		count += sampleSize;
	}
}

void DigitalDistinguish::ForwardPass(Sample& sample) {
	size_t layerCount = layers.size() - 1;
	for (size_t i = 0; i < layerCount; i++) {
		sample.MatrixMultiply(*layers[i], ActiveFunc::Linear);
	}
	sample.MatrixMultiply(*layers[layerCount], ActiveFunc::SoftMax);
}

void DigitalDistinguish::InverseTrans(Sample& sample) {
	vector<SampleLayer>& acLayers = sample.activeLayers;
	acLayers[acLayers.size() - 1].out[sample.m_realValue] -= 1; //变换梯度
	for (int i = acLayers.size() - 2; i > -1; i--) {
		for (int r = 0; r < acLayers[i].out.size(); r++) {
			acLayers[i].out[r] = 0;
			const NeuralMatrix& weightLayer = *layers[i + 1LL];
			for (int wr = 0; wr < weightLayer.row; wr++) {
				acLayers[i].out[r] += weightLayer.matrix[wr][r] * acLayers[i + 1LL].out[wr];
			}
		}
	}
}

int DigitalDistinguish::Distinguish(Sample& sample) {
	ForwardPass(sample);
	double v = -1;
	int result = -1;
	for (int i = 0; i < sample.activeLayers[sample.activeLayers.size() - 1].out.size(); i++) {
		if (v < sample.activeLayers[sample.activeLayers.size() - 1].out[i]) {
			result = i;
			v = sample.activeLayers[sample.activeLayers.size() - 1].out[i];
		}
	}
	return result;
}

void DigitalDistinguish::Test(const std::vector<Sample*>& data) {
	int corectCount = 0;
	for (auto s : data) {
		int num = Distinguish(*s);
		if (num == s->m_realValue) {
			corectCount++;
		}
		s->activeLayers.clear();
	}
	double corectRate = 100.0 * (double)corectCount / data.size() ;
	printf("正确率: %.3f\%\n", corectRate);
}

void DigitalDistinguish::BackwardsPass(const vector<Sample*>& samples, const vector<size_t>& indeces, double lRate, double averageCostVal) {
	vector<NeuralMatrix*> lyGradient;
	for (long long i = 0; i < layers.size(); i++) {
		NeuralMatrix* gradient = new NeuralMatrix(*layers[i], true);
		lyGradient.push_back(gradient);
	}
	for (size_t index : indeces) {
		Sample& sample = *samples[index];
		InverseTrans(sample); //对样本进行逆变换以求梯度
		for (long long i = 0; i < sample.activeLayers.size(); i++) {
			if (i == 0) { //输入层
				for (int r = 0; r < lyGradient[i]->matrix.size(); r++) {
					switch (sample.activeLayers[i].activeFunc) {
					case ActiveFunc::ReLU:
						if (sample.activeLayers[i].net[r] > 0) {
							lyGradient[i]->bias += sample.activeLayers[i].out[r];
							for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
								lyGradient[i]->matrix[r][c] = sample.activeLayers[i].out[r] * sample.m_data[c];
							}
						}
						break;
					case ActiveFunc::SoftMax:
						for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
							lyGradient[i]->matrix[r][c] = sample.activeLayers[i].out[r] * sample.m_data[c];
						}
						lyGradient[i]->bias += sample.activeLayers[i].out[r];
						break;
					case ActiveFunc::Linear:
						for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
							lyGradient[i]->matrix[r][c] = sample.activeLayers[i].out[r] * sample.m_data[c];
						}
						lyGradient[i]->bias += sample.activeLayers[i].out[r];
						break;
					default:
						break;
					}
				}
			}
			else {
				for (int r = 0; r < lyGradient[i]->matrix.size(); r++) {
					switch (sample.activeLayers[i].activeFunc) {
					case ActiveFunc::ReLU:
						if (sample.activeLayers[i].net[r] > 0) {
							for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
								lyGradient[i]->matrix[r][c] = sample.activeLayers[i].out[r] * sample.activeLayers[i - 1].out[c];
							}
							lyGradient[i]->bias += sample.activeLayers[i].out[r];
						}
						break;
					case ActiveFunc::SoftMax:
						for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
							lyGradient[i]->matrix[r][c] = sample.activeLayers[i].out[r] * sample.activeLayers[i - 1].out[c];
						}
						lyGradient[i]->bias += sample.activeLayers[i].out[r];
						break;
					case ActiveFunc::Linear:
						for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
							lyGradient[i]->matrix[r][c] = sample.activeLayers[i].out[r] * sample.activeLayers[i - 1].out[c];
						}
						lyGradient[i]->bias += sample.activeLayers[i].out[r];
					default:
						break;
					}
				}
			}
		}
		sample.activeLayers.clear();
	}
	for (long long i = 0; i < layers.size(); i++) {
		lyGradient[i]->bias /= indeces.size();
		(layers[i])->bias -= lRate * lyGradient[i]->bias;
		for (int r = 0; r < lyGradient[i]->matrix.size(); r++) {
			for (int c = 0; c < lyGradient[i]->matrix[r].size(); c++) {
				lyGradient[i]->matrix[r][c] /= indeces.size();
				(layers[i])->matrix[r][c] -= lRate * lyGradient[i]->matrix[r][c];
			}
		}
		delete lyGradient[i];
	}
}

DigitalDistinguish::DigitalDistinguish() {}

DigitalDistinguish::~DigitalDistinguish() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];
	}
}