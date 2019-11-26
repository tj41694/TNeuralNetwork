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

//代价函数计算方式
enum class CostFunc {
	//均方差
	MeanSquare,
	//交叉熵
	CrossEntropy
};


struct SampleLayer {
	std::vector<double> net;
	std::vector<double> out;
	ActiveFunc activeFunc;
};


class Sample {
public:
	int								trueValue;
	std::vector<double>				originLayer;
	std::vector<SampleLayer>		activeLayers;
public:

	Sample();
	Sample(const Sample&);
	Sample(int num_, const float* data, unsigned int floatCount);
	~Sample();

	void MatrixMultiply(const NeuralMatrix& layer, ActiveFunc func);
	double GetCostValue(CostFunc func);

	Sample& operator=(const Sample& ly);

private:
	double	costVal = 0;
	bool	costValValid = false;
};

