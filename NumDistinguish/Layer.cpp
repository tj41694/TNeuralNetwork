#include "Layer.h"
#include <corecrt_memory.h>

Layer::Layer() {
	activation = nullptr;
	size = 0;
	trueValue = -1;
}

Layer::Layer(int num_, const char* data, unsigned int dataCount) {
	size = dataCount / sizeof(float);
	activation = new float[size];
	memcpy(activation, data, dataCount);
	trueValue = num_;
}

Layer Layer::MatrixMultiply(Layer& leftLayer) {
	return Layer();
}

Layer::~Layer() {
	if (activation) {
		delete[] activation;
	}
}

