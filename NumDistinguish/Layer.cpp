#include "Layer.h"
#include <corecrt_memory.h>

Layer::Layer() {}

Layer::Layer(const Layer& layer_) {
	activation = layer_.activation;
	size = layer_.size;
	trueValue = layer_.trueValue;
}

Layer::Layer(Layer&& layer_) {
	activation = layer_.activation;
	layer_.activation = nullptr;
	size = layer_.size;
	trueValue = layer_.trueValue;
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
	delete[] activation;
}

