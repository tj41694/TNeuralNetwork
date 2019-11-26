#include "Sample.h"
#include <corecrt_memory.h>
#include "NeuralMatrix.h"

Sample::Sample() {
	memset(&originLayer, 0, sizeof(SampleUnit));
	trueValue = 0;
}

Sample::Sample(const Sample& sample_) {
	originLayer.activation = new float[sample_.originLayer.size];
	memcpy(originLayer.activation, sample_.originLayer.activation, sample_.originLayer.size * sizeof(float));
	originLayer.size = sample_.originLayer.size;
	trueValue = sample_.trueValue;
}

Sample::Sample(int num_, const char* data, unsigned int dataCount) {
	originLayer.size = dataCount / sizeof(float);
	originLayer.activation = new float[originLayer.size];
	memcpy(originLayer.activation, data, dataCount);
	trueValue = num_;
}

void Sample::MatrixMultiply(const NeuralMatrix& neuralMat, ActiveFunc func) {

	SampleUnit lastLayer;

	activeLayers.size() == 0 ? lastLayer = originLayer : lastLayer = activeLayers[activeLayers.size() - 1].out;

	if (lastLayer.size != neuralMat.column) { printf("err.. Dimension not match..\n"); return; }

	SampleLayer layer{ 0 };
	layer.activeFunc = func;
	layer.net.activation = new float[neuralMat.row];
	layer.net.size = neuralMat.row;
	layer.out.activation = new float[neuralMat.row];
	layer.out.size = neuralMat.row;
	for (int r = 0; r < neuralMat.row; r++) { //矩阵乘数据(向量)
		float val = 0;
		for (int c = 0; c < neuralMat.column; c++) {
			val += lastLayer.activation[c] * neuralMat.matrix[r * neuralMat.column + c];
		}
		layer.net.activation[r] = val + neuralMat.bias;
	}
	switch (layer.activeFunc) {
	case ActiveFunc::ReLU:
		for (unsigned int i = 0; i < layer.net.size; i++) {
			layer.net.activation[i] > 0 ?
				layer.out.activation[i] = layer.net.activation[i] :
				layer.out.activation[i] = 0;
		}
		break;
	case ActiveFunc::SoftMax:
	{
		double total = 0; //先求和
		for (unsigned int i = 0; i < layer.net.size; i++) {
			//printf("ori: %f\n", layer.net.activation[i]);
			layer.out.activation[i] = (float)exp(layer.net.activation[i]);
			total += layer.out.activation[i];
		}
		for (unsigned int i = 0; i < layer.net.size; i++) { //再赋值
			layer.out.activation[i] = (float)(layer.out.activation[i] / total);
			//printf("softmax: %f\n", layer.out.activation[i]);
		}
	}
	break;
	case ActiveFunc::Linear:
	default:
		for (unsigned int i = 0; i < layer.net.size; i++) {
			layer.out.activation[i] = layer.net.activation[i];
		}
		break;
	}
	activeLayers.push_back(layer);
}

double Sample::GetCostValue(CostFunc func) {
	if (!costValValid) {
		SampleUnit outputLayer = activeLayers[activeLayers.size() - 1].out;
		switch (func) {
		case CostFunc::CrossEntropy:
			costVal = -log(outputLayer.activation[trueValue]);
			break;
		case CostFunc::MeanSquare:
		default:
			for (unsigned int i = 0; i < outputLayer.size; i++) {
				if (i == trueValue) {
					costVal += (outputLayer.activation[i] - 1.0) * (outputLayer.activation[i] - 1.0);
				}
				else {
					costVal += outputLayer.activation[i] * (double)outputLayer.activation[i];
				}
			}
			break;
		}
	}
	return costVal;
}

Sample& Sample::operator=(const Sample& sample_) {
	if (this == &sample_) { return *this; }
	delete[] originLayer.activation;
	originLayer.activation = new float[sample_.originLayer.size];
	memcpy(originLayer.activation, sample_.originLayer.activation, sample_.originLayer.size * sizeof(float));
	originLayer.size = sample_.originLayer.size;
	trueValue = sample_.trueValue;
	return *this;
}

Sample::~Sample() {
	if (originLayer.activation) {
		delete[] originLayer.activation;
	}
}