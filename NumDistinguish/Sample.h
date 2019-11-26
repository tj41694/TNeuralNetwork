#pragma once
#include <math.h>
#include <stdio.h>
#include <vector>

class NeuralMatrix;

enum class ActiveFunc {
	Linear,
	ReLU,
	SoftMax
};

//���ۺ������㷽ʽ
enum class CostFunc {
	//������
	MeanSquare,
	//������
	CrossEntropy
};

struct SampleUnit {
	float*			activation;
	unsigned int	size;
};

struct SampleLayer {
	SampleUnit net;
	SampleUnit out;
	ActiveFunc activeFunc;
};


class Sample {
public:
	int							trueValue;
	SampleUnit					originLayer;
	std::vector<SampleLayer>		activeLayers;
public:

	Sample();
	Sample(const Sample&);
	Sample(int num_, const char* data, unsigned int dataCount);
	~Sample();

	void MatrixMultiply(const NeuralMatrix& layer, ActiveFunc func);
	double GetCostValue(CostFunc func);

	Sample& operator=(const Sample& ly);

private:
	double	costVal = 0;
	bool	costValValid = false;
};

