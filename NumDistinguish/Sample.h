#pragma once
#include <math.h>
class Sample {
public:
	float*			activation;
	unsigned int	size;
	int				trueValue;

public:
	Sample();
	Sample(const Sample& ly);
	Sample(int num_, const char* data, unsigned int dataCount);
	void Relu() {
		for (unsigned int i = 0; i < size; i++) {
			if (activation[i] < 0) {
				activation[i] = 0;
			}
		}
	}
	Sample& operator=(const Sample& ly);
	//经典SoftMax化
	Sample SoftMax() {
		Sample result;
		result.activation = new float[size] { 0 };
		result.size = size;
		result.trueValue = trueValue;
		//先求和
		double total = 0;
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
	~Sample();
private:
};

