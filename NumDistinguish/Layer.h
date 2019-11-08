#pragma once
#include <math.h>
class Layer {
public:
	float*			activation = 0;
	unsigned int	size = 0;
	int				trueValue = -1;

public:
	Layer();
	Layer(const Layer& layer_);
	Layer(Layer&& layer_);
	Layer(int num_, const char* data, unsigned int dataCount);
	Layer MatrixMultiply(Layer& leftLayer);
	void Relu() {
		for (unsigned int i = 0; i < size; i++) {
			if (activation[i] < 0) {
				activation[i] = 0;
			}
		}
	}
	//经典SoftMax化
	Layer SoftMax() {
		//先求和
		double total = 0;
		Layer result;
		result.activation = new float[size] { 0 };
		for (unsigned int i = 0; i < size; i++) {
			//printf("ori: %f\n", pixs[i]);
			result.activation[i] = (float)exp(activation[i]);
			total += result.activation[i];
		}
		//再赋值
		for (unsigned int i = 0; i < size; i++) {
			result.activation[i] = (float)(result.activation[i] / total);
			//printf("softmax: %f\n", result.pixs[i]);
		}
		return result;
	}
	~Layer();
private:
};

