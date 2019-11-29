#include "Sample.h"
#include <corecrt_memory.h>
#include "NeuralMatrix.h"

using namespace std;
Sample::Sample() {
	trueValue = 0;
}

Sample::Sample(const Sample& sample_) {
	originLayer = sample_.originLayer;
	trueValue = sample_.trueValue;
}

Sample::Sample(int num_, const float* data, unsigned int floatCount) {
	originLayer.resize(floatCount);
	for (unsigned int i = 0; i < floatCount; i++) {
		originLayer[i] = data[i];
	}
	trueValue = num_;
}

void Sample::MatrixMultiply(const NeuralMatrix& neuralMat, ActiveFunc func) {

	vector<double>* lastActiveLayer;

	activeLayers.size() == 0 ? lastActiveLayer = &originLayer : lastActiveLayer = &activeLayers[activeLayers.size() - 1].out;

	if (lastActiveLayer->size() != neuralMat.column) { printf("err.. Dimension not match..\n"); return; }

	SampleLayer layer;
	layer.activeFunc = func;

	layer.net.resize(neuralMat.row);
	layer.out.resize(neuralMat.row);
	for (int r = 0; r < neuralMat.row; r++) { //矩阵乘数据(向量)
		double val = 0;
		for (int c = 0; c < neuralMat.column; c++) {
			val += (*lastActiveLayer)[c] * neuralMat.matrix[r][c];
		}
		layer.net[r] = val + neuralMat.bias;
	}
	switch (layer.activeFunc) {
	case ActiveFunc::ReLU:
		for (unsigned int i = 0; i < layer.net.size(); i++) {
			layer.net[i] > 0 ?
				layer.out[i] = layer.net[i] :
				layer.out[i] = 0;
		}
		break;
	case ActiveFunc::SoftMax:
	{
		double total = 0; //先求和
		for (unsigned int i = 0; i < layer.net.size(); i++) {
			//printf("ori: %f\n", layer.net.activation[i]);
			layer.out[i] = exp(layer.net[i]);
			total += layer.out[i];
		}
		for (unsigned int i = 0; i < layer.net.size(); i++) { //再赋值
			layer.out[i] = layer.out[i] / total;
			//printf("softmax: %f\n", layer.out.activation[i]);
		}
	}
	break;
	case ActiveFunc::Linear:
	default:
		for (unsigned int i = 0; i < layer.net.size(); i++) {
			layer.out[i] = layer.net[i];
		}
		break;
	}
	activeLayers.push_back(layer);
}

double Sample::GetCostValue(CostFunc func) {
	if (!costValValid) {
		const vector<double> & outputLayer = activeLayers[activeLayers.size() - 1].out;
		switch (func) {
		case CostFunc::CrossEntropy:
			costVal = -log(outputLayer[trueValue]);
			break;
		case CostFunc::MeanSquare:
		default:
			for (unsigned int i = 0; i < outputLayer.size(); i++) {
				if (i == trueValue) {
					costVal += (outputLayer[i] - 1.0) * (outputLayer[i] - 1.0);
				}
				else {
					costVal += outputLayer[i] * (double)outputLayer[i];
				}
			}
			break;
		}
		costValValid = true;
	}
	return costVal;
}

Sample& Sample::operator=(const Sample& sample_) {
	if (this == &sample_) { return *this; }
	originLayer = sample_.originLayer;
	activeLayers = sample_.activeLayers;
	trueValue = sample_.trueValue;
	return *this;
}

Sample::~Sample() {
}