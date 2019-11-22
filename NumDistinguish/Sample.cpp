#include "Sample.h"
#include <corecrt_memory.h>

Sample::Sample() {
	activation = nullptr;
	size = 0;
	trueValue = -1;
}

Sample::Sample(const Sample& ly) {
	activation = new float[ly.size];
	memcpy(activation, ly.activation, ly.size * sizeof(float));
	size = ly.size;
	trueValue = ly.trueValue;
}

Sample::Sample(int num_, const char* data, unsigned int dataCount) {
	size = dataCount / sizeof(float);
	activation = new float[size];
	memcpy(activation, data, dataCount);
	trueValue = num_;
}

Sample& Sample::operator=(const Sample& ly) {
	if (this == &ly) { return *this; }
	delete[] activation;
	activation = new float[ly.size];
	memcpy(activation, ly.activation, ly.size * sizeof(float));
	size = ly.size;
	trueValue = ly.trueValue;
	return *this;
}

Sample::~Sample() {
	if (activation) {
		delete[] activation;
	}
}

